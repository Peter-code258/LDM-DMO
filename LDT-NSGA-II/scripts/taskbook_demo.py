# -*- coding: utf-8 -*-
"""任务书配套示例：两段可直接运行的代码。

片段 A —— 分布迁移核心（只依赖 numpy，独立可跑，演示三个公式）
片段 B —— 端到端调用本项目真实 API（跑一次 FDA1 并报告稳态 MIGD）

运行（任选一种）：
    D:\\python\\python.exe -m scripts.taskbook_demo          # 在 LDT-NSGA-II 目录下
    set PYTHONIOENCODING=utf-8 && D:\\python\\python.exe scripts\\taskbook_demo.py
"""
import sys

import numpy as np

# Windows 控制台默认可能是 GBK 代码页，中文会乱码；显式改用 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# ──────────────────── 片段 A：分布迁移核心（独立可跑）────────────────────
def distribution_transfer_demo(seed=0, D=5, N=100, alpha=0.5):
    """演示 LDT 的三个核心公式，只用 numpy。

    返回 (mu_e, sigma_e, mu_hat, X_new_norm)。
    """
    rng = np.random.default_rng(seed)

    # 上一环境末种群归一化到 [0,1]^D（真实代码里由 normalize() 完成）
    Z = np.clip(rng.normal(0.5, 0.15, size=(N, D)), 0.0, 1.0)
    # 取"非支配子集"的占位：这里用全部个体（真实代码用非支配筛选 + 回退规则）
    ND = Z

    # 公式 1：逐维均值与逐维标准差（归一化单位下限 1e-3）
    mu_e = ND.mean(axis=0)
    sigma_e = np.maximum(ND.std(axis=0, ddof=0), 1e-3)

    # 公式 2：趋势中心外推 mu_hat = clip(mu_e + alpha*(mu_e - mu_prev), 0, 1)
    mu_prev = mu_e - 0.05                      # 假设上一环境中心偏左 0.05
    mu_hat = np.clip(mu_e + alpha * (mu_e - mu_prev), 0.0, 1.0)

    # 公式 3：迁移个体 x = clip(mu_hat + z .* sigma_e, 0, 1)，z ~ N(0, I)
    z = rng.normal(0.0, 1.0, size=(N, D))
    X_new_norm = np.clip(mu_hat[None, :] + z * sigma_e[None, :], 0.0, 1.0)
    return mu_e, sigma_e, mu_hat, X_new_norm


# ──────────────────── 片段 B：端到端调用真实 API ────────────────────
def end_to_end_demo(seed=0):
    """调用项目的真实接口跑一次 FDA1/τ=25，返回稳态 MIGD。"""
    from collections import defaultdict

    from experiments.common import RunConfig, run_single

    cfg = RunConfig(problem="FDA1", tau_t=25, n_t=20, N=100, n_gen=300,
                    algorithms=["ldt_nsga2"], code_version="v3.2",
                    experiment_group="taskbook_demo")
    rows = run_single("ldt_nsga2", "FDA1", cfg, seed)

    # 稳态 MIGD：每个环境段最后一代 IGD 的段间平均
    by_env = defaultdict(dict)
    for r in rows:
        by_env[int(r["environment_index"])][int(r["generation"])] = float(r["IGD"])
    steady = float(np.mean([by_env[e][max(by_env[e])] for e in sorted(by_env)]))
    return len(rows), len(by_env), steady


if __name__ == "__main__":
    mu_e, sigma_e, mu_hat, X = distribution_transfer_demo()
    print("片段 A：分布迁移核心")
    print("  mu_e    =", np.round(mu_e, 4))
    print("  sigma_e =", np.round(sigma_e, 4))
    print("  mu_hat  =", np.round(mu_hat, 4), "  (alpha=0.5，应比 mu_e 略靠右)")
    print("  迁移个体形状 =", X.shape, " 取值范围 =",
          f"[{X.min():.4f}, {X.max():.4f}]")
    print("  逐维均值（迁移后）= ", np.round(X.mean(axis=0), 4))

    n_rows, n_envs, steady = end_to_end_demo()
    print()
    print("片段 B：端到端（FDA1, tau=25, seed=0）")
    print("  逐代记录行数 =", n_rows, " 环境段数 =", n_envs)
    print("  稳态 MIGD =", f"{steady:.5f}")
