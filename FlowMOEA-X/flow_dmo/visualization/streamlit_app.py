"""Streamlit real-time dual-space (Decision Manifold PS vs Objective Front PF)
interactive workbench (Section 8.2).

Launch via:
    streamlit run flow_dmo/visualization/streamlit_app.py
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from flow_dmo.algorithms.flow_moea_x import FlowMOEAX
from flow_dmo.config import TransportConfig, get_device
from flow_dmo.core.driver import BatchedDynamicDriver
from flow_dmo.core.tensor_ops import denormalize
from flow_dmo.problems import ALL_ANALYTICAL_PROBLEMS, make_problem


def main() -> None:
    st.set_page_config(page_title="FlowMOEA-X Dual-Space Manifold Player", layout="wide")
    st.title("FlowMOEA-X v4.0 — Real-Time Dual-Space Manifold Workbench")

    with st.sidebar:
        st.header("Experiment Controls")
        problem_name = st.selectbox("Benchmark Problem", ALL_ANALYTICAL_PROBLEMS + ["Microgrid-DEED"], index=9)
        kernel_name = st.selectbox("Evolution Kernel", ["nsga2", "nsga3", "moead"], index=0)
        mode = st.selectbox("FlowMOEA-X Mode", ["lite", "pro"], index=0)
        tau_t = st.slider("Change Frequency (tau_t)", 3, 25, 8)
        n_t = st.slider("Change Severity (n_t)", 5, 20, 10)
        n_envs = st.slider("Environments", 4, 20, 8)
        pop_size = st.slider("Population Size (N)", 30, 120, 60, step=10)

        st.subheader("Module Ablation Toggles")
        use_sliced_ot = st.checkbox("Sliced-OT Manifold Push-Forward", value=True)
        use_kalman = st.checkbox("Per-Dimension Kalman Momentum", value=True)
        use_krr = st.checkbox("KRR 4x Zero-FE Surrogate Pre-Screening", value=True)
        run_btn = st.button("Run Simulation", type="primary")

    if run_btn or "traj" not in st.session_state:
        dev = get_device("auto")
        prob = make_problem(problem_name, tau_t=tau_t, n_t=n_t, device=dev)
        tcfg = TransportConfig(
            mode=mode,
            use_sliced_ot=use_sliced_ot,
            use_kalman=use_kalman,
            use_krr=use_krr,
        )
        algo = FlowMOEAX(pop_size=pop_size, n_var=prob.n_var, n_obj=prob.n_obj, device=dev, transport_cfg=tcfg)
        driver = BatchedDynamicDriver(
            problem=prob, algorithm=algo, kernel=kernel_name, n_seeds=3, pop_size=pop_size,
            tau_t=tau_t, n_envs=n_envs, seed=2026, strict_fe_budget=True, n_pf_points=250,
        )
        with st.spinner("Running batched PyTorch evolution..."):
            st.session_state["traj"] = driver.run()
            st.session_state["prob"] = prob

    traj = st.session_state["traj"]
    prob = st.session_state["prob"]
    env_idx = st.slider("Environment Step k", 0, traj.n_envs - 1, traj.n_envs - 1)
    t_val = traj.times[env_idx]

    col_ps, col_pf = st.columns(2)
    with col_ps:
        st.subheader(f"Decision Space Manifold (PS) — t = {t_val:.2f}")
        Z_env = traj.Z_env_end[env_idx, 0].cpu()
        X_env = denormalize(Z_env.unsqueeze(0), prob.xl.cpu(), prob.xu.cpu())[0].numpy()
        fig_ps = go.Figure()
        try:
            X_ps = prob.get_true_ps(t_val, n_points=200, device="cpu").numpy()
            fig_ps.add_trace(go.Scatter(x=X_ps[:, 0], y=X_ps[:, 1], mode="lines", name="True PS",
                                        line=dict(color="#0f172a", width=2.5)))
        except NotImplementedError:
            pass
        fig_ps.add_trace(go.Scatter(x=X_env[:, 0], y=X_env[:, 1], mode="markers", name="FlowMOEA-X Pop",
                                    marker=dict(color="#0ea5e9", size=7, opacity=0.85)))
        # Legacy Gaussian 1-sigma ellipse for visual comparison
        mu = X_env[:, :2].mean(axis=0)
        sd = X_env[:, :2].std(axis=0)
        ang = np.linspace(0, 2 * np.pi, 80)
        fig_ps.add_trace(go.Scatter(x=mu[0] + 1.5 * sd[0] * np.cos(ang), y=mu[1] + 1.5 * sd[1] * np.sin(ang),
                                    mode="lines", name="Isotropic Gaussian (Legacy)",
                                    line=dict(color="#6366f1", dash="dash")))
        fig_ps.update_layout(xaxis_title="x_1", yaxis_title="x_2", height=420, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_ps, use_container_width=True)

    with col_pf:
        st.subheader(f"Objective Space Front (PF) — t = {t_val:.2f}")
        F_env = traj.F_env_end[env_idx, 0].cpu().numpy()
        PF_true = traj.PF_true[env_idx].cpu().numpy()
        fig_pf = go.Figure()
        if prob.n_obj == 2:
            fig_pf.add_trace(go.Scatter(x=PF_true[:, 0], y=PF_true[:, 1], mode="lines", name="True PF",
                                        line=dict(color="#0f172a", width=2.5)))
            fig_pf.add_trace(go.Scatter(x=F_env[:, 0], y=F_env[:, 1], mode="markers", name="FlowMOEA-X",
                                        marker=dict(color="#0ea5e9", size=7)))
            fig_pf.update_layout(xaxis_title="f_1", yaxis_title="f_2", height=420, margin=dict(l=20, r=20, t=20, b=20))
        else:
            fig_pf.add_trace(go.Scatter3d(x=PF_true[:, 0], y=PF_true[:, 1], z=PF_true[:, 2], mode="markers",
                                          name="True PF", marker=dict(size=2, color="#64748b", opacity=0.4)))
            fig_pf.add_trace(go.Scatter3d(x=F_env[:, 0], y=F_env[:, 1], z=F_env[:, 2], mode="markers",
                                          name="FlowMOEA-X", marker=dict(size=4, color="#0ea5e9")))
            fig_pf.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pf, use_container_width=True)

    # Bottom scrolling IGD shock-wave curve
    st.subheader("Per-Generation IGD Recovery Shock-Wave")
    igd = traj.igd_per_gen.cpu().numpy()
    fig_igd = go.Figure()
    fig_igd.add_trace(go.Scatter(y=np.median(igd, axis=1), mode="lines", name="Median IGD",
                                 line=dict(color="#0ea5e9", width=2.5)))
    fig_igd.update_yaxes(type="log", title="IGD (log)")
    fig_igd.update_xaxes(title="Generation")
    fig_igd.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig_igd, use_container_width=True)


if __name__ == "__main__":
    main()
