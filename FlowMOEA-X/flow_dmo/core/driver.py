"""Full-tensor batched dynamic evolution driver (Section 2 & 7).

Runs all ``B`` independent random seeds simultaneously in 3-D tensors
``(B, N, D)`` across ``n_envs`` environments, recording:
  * end-of-environment objective tensors ``F_env_end (T_envs, B, N, M)``
  * warm-start (gen-0) objective tensors ``F_warmstart (T_envs, B, N, M)``
  * per-generation IGD traces ``igd_per_gen (T_envs * tau_t, B)``
  * exact ``FELedger`` summary and per-switch response latencies (ms).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from flow_dmo.algorithms.base import BatchedDynamicMOEA
from flow_dmo.config import make_generator
from flow_dmo.core.fe_ledger import FELedger
from flow_dmo.core.kernels import EvolutionKernel, make_kernel
from flow_dmo.core.tensor_ops import uniform_population
from flow_dmo.evaluation.gpu_metrics import batched_igd
from flow_dmo.problems.base import DynamicProblem
from flow_dmo.transport.change_detector import PopulationChangeDetector

Tensor = torch.Tensor


@dataclass
class RunTrajectory:
    problem_name: str
    algorithm_name: str
    kernel_name: str
    n_seeds: int
    pop_size: int
    tau_t: int
    n_t: int
    n_envs: int
    times: list[float]
    F_env_end: Tensor            # (T_envs, B, N, M)
    F_warmstart: Tensor          # (T_envs, B, N, M)
    Z_env_end: Tensor            # (T_envs, B, N, D)
    PF_true: list[Tensor]        # list of (P_t, M)
    igd_per_gen: Tensor          # (T_envs * tau_t, B)
    transfer_times_ms: list[float]
    total_wall_s: float
    ledger_summary: dict
    alpha_history: list[Tensor] = field(default_factory=list)


class BatchedDynamicDriver:
    """Orchestrates synchronised 3-D batched dynamic multi-objective runs."""

    def __init__(
        self,
        problem: DynamicProblem,
        algorithm: BatchedDynamicMOEA,
        kernel: EvolutionKernel | str = "nsga2",
        n_seeds: int = 30,
        pop_size: int = 100,
        tau_t: int = 10,
        n_envs: int = 20,
        seed: int = 2026,
        strict_fe_budget: bool = True,
        eps_change: float = 1e-4,
        n_pf_points: int = 500,
    ):
        self.problem = problem
        self.algorithm = algorithm
        self.n_seeds = int(n_seeds)
        self.pop_size = int(pop_size)
        self.tau_t = int(tau_t)
        self.n_envs = int(n_envs)
        self.seed = int(seed)
        self.strict_fe_budget = bool(strict_fe_budget)
        self.n_pf_points = int(n_pf_points)
        self.kernel = (
            make_kernel(kernel, pop_size=self.pop_size, n_obj=problem.n_obj)
            if isinstance(kernel, str)
            else kernel
        )
        self.detector = PopulationChangeDetector(eps_change=eps_change)

    def run(self) -> RunTrajectory:
        dev = self.problem.device
        dt = self.problem.dtype
        B, N, D, M = self.n_seeds, self.pop_size, self.problem.n_var, self.problem.n_obj
        gen = make_generator(self.seed, dev)
        self.algorithm.reset()

        ledger = FELedger(
            n_seeds=B,
            pop_size=N,
            tau_t=self.tau_t,
            device=dev,
            strict_fe_budget=self.strict_fe_budget,
        )

        F_end_list: list[Tensor] = []
        Z_end_list: list[Tensor] = []
        F_warm_list: list[Tensor] = []
        PF_list: list[Tensor] = []
        igd_gen_rows: list[Tensor] = []
        times: list[float] = []
        transfer_ms: list[float] = []

        t_wall0 = time.perf_counter()
        Z = uniform_population(B, N, D, device=dev, dtype=dt, generator=gen)
        F: Tensor | None = None

        for env in range(self.n_envs):
            t = self.problem.time_from_step(env)
            times.append(t)
            PF_t = self.problem.get_true_pf(t, n_points=self.n_pf_points, device=dev, dtype=dt)
            PF_list.append(PF_t)
            ledger.start_environment(env, t)

            if env == 0:
                F = ledger.evaluate_and_charge(self.problem, Z, t, role="warmstart")
            else:
                assert F is not None
                det = self.detector.detect(Z, F, self.problem, t_new=t, ledger=ledger)
                t_tr0 = time.perf_counter()
                Z, F = self.algorithm.respond_batch(
                    prev_Z=Z,
                    prev_F=F,
                    sensor_Z=det.sensor_Z,
                    sensor_F_new=det.sensor_F_new,
                    D_k=det.D_k,
                    problem=self.problem,
                    t_new=t,
                    ledger=ledger,
                    generator=gen,
                )
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                transfer_ms.append((time.perf_counter() - t_tr0) * 1000.0)

            F_warm_list.append(F.detach().clone())
            igd_gen_rows.append(batched_igd(F, PF_t))

            n_rem = ledger.remaining_generations_in_env()
            for _ in range(n_rem):
                Z, F = self.kernel.step_generation(
                    Z, F, lambda z_off: ledger.evaluate_and_charge(self.problem, z_off, t, role="evolution"), gen
                )
                igd_gen_rows.append(batched_igd(F, PF_t))

            F_end_list.append(F.detach().clone())
            Z_end_list.append(Z.detach().clone())

        total_wall = time.perf_counter() - t_wall0
        alpha_hist = getattr(self.algorithm, "alpha_history", [])

        return RunTrajectory(
            problem_name=self.problem.name,
            algorithm_name=self.algorithm.name,
            kernel_name=self.kernel.name,
            n_seeds=B,
            pop_size=N,
            tau_t=self.tau_t,
            n_t=self.problem.n_t,
            n_envs=self.n_envs,
            times=times,
            F_env_end=torch.stack(F_end_list, dim=0),
            F_warmstart=torch.stack(F_warm_list, dim=0),
            Z_env_end=torch.stack(Z_end_list, dim=0),
            PF_true=PF_list,
            igd_per_gen=torch.stack(igd_gen_rows, dim=0),
            transfer_times_ms=transfer_ms,
            total_wall_s=total_wall,
            ledger_summary=ledger.summary(),
            alpha_history=list(alpha_hist),
        )
