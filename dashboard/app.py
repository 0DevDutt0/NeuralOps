"""NeuralOps Streamlit Dashboard — multi-page LLMOps monitoring UI.

6 pages:
    Overview       — KPI cards and system health
    Prompt Manager — CRUD, version timeline, diff viewer
    Experiments    — A/B test management and live results
    Model Registry — Register and monitor LLM models
    Drift Monitor  — Quality drift timeline and alert history
    Cost Tracker   — Token usage and cost breakdown charts

Run with: streamlit run dashboard/app.py
"""

import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("DASHBOARD_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="NeuralOps",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme via custom CSS ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #313244;
    }
    .stMetric { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Prompt Manager", "Experiments", "Model Registry", "Drift Monitor", "Cost Tracker"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"API: `{API_URL}`")

# ── Shared helpers ────────────────────────────────────────────────────────────


@st.cache_data(ttl=60)
def api_get(path: str) -> dict | list | None:
    """Fetch from the NeuralOps API with caching.

    Args:
        path: URL path e.g. "/health".

    Returns:
        Parsed JSON response, or None on error.
    """
    try:
        resp = httpx.get(f"{API_URL}{path}", timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API error ({path}): {exc}")
        return None


def api_post(path: str, data: dict) -> dict | None:
    """POST to the API without caching.

    Args:
        path: URL path.
        data: JSON body.

    Returns:
        Parsed JSON response, or None on error.
    """
    try:
        resp = httpx.post(f"{API_URL}{path}", json=data, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("🧠 NeuralOps — Overview")

    health = api_get("/health/")
    if health:
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", health.get("status", "?").upper())
        col2.metric("Environment", health.get("environment", "?"))
        col3.metric("LLM Provider", health.get("llm_provider", "?"))

    ready = api_get("/health/ready")
    if ready:
        db_ok = ready.get("database") == "ok"
        st.success("Database: connected") if db_ok else st.error("Database: disconnected")

    st.markdown("---")
    st.subheader("Platform Snapshot")

    col1, col2, col3, col4 = st.columns(4)

    prompts = api_get("/api/v1/prompts/") or []
    col1.metric("Total Prompts", len(prompts))

    experiments = api_get("/api/v1/experiments/") or []
    running = sum(1 for e in experiments if e.get("status") == "running")
    col2.metric("Running Experiments", running)

    models_list = api_get("/api/v1/models/") or []
    col3.metric("Registered Models", len(models_list))

    drift_summary = api_get("/api/v1/drift/summary")
    if drift_summary:
        col4.metric(
            "Drifting Prompts",
            drift_summary.get("drifting_prompts", 0),
            delta=f"/{drift_summary.get('total_prompts', 0)} total",
            delta_color="inverse",
        )

    # Experiment status breakdown
    if experiments:
        st.subheader("Experiment Status Breakdown")
        status_counts = pd.Series([e.get("status", "unknown") for e in experiments]).value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            color_discrete_sequence=px.colors.sequential.Plasma_r,
        )
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Prompt Manager
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Prompt Manager":
    st.title("📝 Prompt Manager")

    tab1, tab2 = st.tabs(["Browse Prompts", "Create Prompt"])

    with tab1:
        prompts = api_get("/api/v1/prompts/") or []
        if not prompts:
            st.info("No prompts yet. Create one in the 'Create Prompt' tab.")
        else:
            prompt_names = {p["name"]: p["id"] for p in prompts}
            selected_name = st.selectbox("Select prompt", list(prompt_names.keys()))

            if selected_name:
                prompt_id = prompt_names[selected_name]
                detail = api_get(f"/api/v1/prompts/{prompt_id}")

                if detail:
                    st.caption(f"ID: `{prompt_id}`")
                    if detail.get("description"):
                        st.markdown(f"**Description:** {detail['description']}")

                    versions = detail.get("versions", [])
                    if versions:
                        st.subheader(f"Versions ({len(versions)})")
                        df = pd.DataFrame([
                            {
                                "Version": v["version"],
                                "Active": "✅" if v["is_active"] else "",
                                "Created": v["created_at"][:10],
                                "Content Preview": v["content"][:60] + "..." if len(v["content"]) > 60 else v["content"],
                            }
                            for v in versions
                        ])
                        st.dataframe(df, use_container_width=True)

                        # Diff viewer
                        st.subheader("Version Diff")
                        version_tags = [v["version"] for v in versions]
                        if len(version_tags) >= 2:
                            col1, col2 = st.columns(2)
                            v1 = col1.selectbox("From", version_tags, index=0)
                            v2 = col2.selectbox("To", version_tags, index=min(1, len(version_tags) - 1))
                            if st.button("Compute Diff") and v1 != v2:
                                diff_data = api_get(f"/api/v1/prompts/{prompt_id}/diff/{v1}/{v2}")
                                if diff_data:
                                    st.metric("Additions", diff_data["additions"])
                                    st.metric("Deletions", diff_data["deletions"])
                                    st.code(diff_data["diff"], language="diff")
                    else:
                        st.info("No versions yet. Add one below.")

                    # Add version form
                    st.subheader("Add New Version")
                    with st.form("add_version"):
                        ver_str = st.text_input("Version (semver)", placeholder="1.0.0")
                        content = st.text_area("Prompt Content", height=150)
                        system_prompt = st.text_area("System Prompt (optional)", height=80)
                        submitted = st.form_submit_button("Create Version")
                        if submitted and ver_str and content:
                            result = api_post(
                                f"/api/v1/prompts/{prompt_id}/versions/",
                                {"version": ver_str, "content": content, "system_prompt": system_prompt or None},
                            )
                            if result:
                                st.success(f"Version {ver_str} created!")
                                st.cache_data.clear()
                                st.rerun()

    with tab2:
        st.subheader("Create New Prompt")
        with st.form("create_prompt"):
            name = st.text_input("Prompt Name", placeholder="my-assistant-prompt")
            description = st.text_area("Description (optional)")
            submitted = st.form_submit_button("Create Prompt")
            if submitted and name:
                result = api_post("/api/v1/prompts/", {"name": name, "description": description or None})
                if result:
                    st.success(f"Prompt '{name}' created with ID: `{result['id']}`")
                    st.cache_data.clear()
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Experiments
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Experiments":
    st.title("🧪 A/B Experiments")

    tab1, tab2 = st.tabs(["Active Experiments", "Create Experiment"])

    with tab1:
        experiments = api_get("/api/v1/experiments/") or []
        if not experiments:
            st.info("No experiments yet.")
        else:
            for exp in experiments:
                with st.expander(
                    f"{'✅' if exp['status'] == 'completed' else '🔄'} {exp['name']} — {exp['status'].upper()}"
                ):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Version A", exp["version_a"])
                    col2.metric("Version B", exp["version_b"])
                    col3.metric("Mean Score A", round(exp["mean_score_a"], 2))
                    col4.metric("Mean Score B", round(exp["mean_score_b"], 2))

                    if exp.get("winner"):
                        st.success(f"🏆 Winner: Version {exp['winner']}")

                    st.caption(f"Trials: {exp['trial_count']} | ID: `{exp['id']}`")

                    # Significance check
                    if st.button("Check Significance", key=f"sig_{exp['id']}"):
                        sig = api_get(f"/api/v1/experiments/{exp['id']}/significance")
                        if sig:
                            cols = st.columns(3)
                            cols[0].metric("p-value", sig.get("p_value") or "N/A")
                            cols[1].metric("Significant?", "Yes" if sig["is_significant"] else "No")
                            cols[2].metric("Winner", sig.get("winner") or "—")

                    # Run a trial
                    st.subheader("Run Trial")
                    user_input = st.text_area("User Input", key=f"input_{exp['id']}")
                    if st.button("Run", key=f"run_{exp['id']}") and user_input:
                        with st.spinner("Running trial..."):
                            result = api_post(
                                f"/api/v1/experiments/{exp['id']}/trials/",
                                {"user_input": user_input},
                            )
                        if result:
                            col1, col2 = st.columns(2)
                            col1.markdown(f"**Version A** (score: {result['score_a']:.2f})\n\n{result['output_a']}")
                            col2.markdown(f"**Version B** (score: {result['score_b']:.2f})\n\n{result['output_b']}")

    with tab2:
        st.subheader("Create A/B Experiment")
        prompts = api_get("/api/v1/prompts/") or []
        if not prompts:
            st.warning("Create a prompt with at least 2 versions first.")
        else:
            with st.form("create_experiment"):
                prompt_options = {p["name"]: p["id"] for p in prompts}
                selected_prompt = st.selectbox("Prompt", list(prompt_options.keys()))
                exp_name = st.text_input("Experiment Name")
                col1, col2 = st.columns(2)
                version_a = col1.text_input("Version A", placeholder="1.0.0")
                version_b = col2.text_input("Version B", placeholder="2.0.0")
                submitted = st.form_submit_button("Create Experiment")
                if submitted and exp_name and version_a and version_b:
                    result = api_post("/api/v1/experiments/", {
                        "name": exp_name,
                        "prompt_id": prompt_options[selected_prompt],
                        "version_a": version_a,
                        "version_b": version_b,
                    })
                    if result:
                        st.success(f"Experiment created: `{result['id']}`")
                        st.cache_data.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Registry
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Model Registry":
    st.title("🤖 Model Registry")

    tab1, tab2 = st.tabs(["Registered Models", "Register New Model"])

    with tab1:
        models_data = api_get("/api/v1/models/") or []
        if not models_data:
            st.info("No models registered yet.")
        else:
            df = pd.DataFrame([
                {
                    "Name": m["name"],
                    "Provider": m["provider"],
                    "Context Window": m["context_window"],
                    "Input Cost ($/1M)": m["cost_per_1k_input_tokens"] * 1000,
                    "Output Cost ($/1M)": m["cost_per_1k_output_tokens"] * 1000,
                    "Active": "✅" if m["is_active"] else "❌",
                    "Priority": m["routing_priority"],
                }
                for m in models_data
            ])
            st.dataframe(df, use_container_width=True)

            # Cost comparison chart
            st.subheader("Cost Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Input",
                x=[m["name"] for m in models_data],
                y=[m["cost_per_1k_input_tokens"] * 1000 for m in models_data],
                marker_color="#7c3aed",
            ))
            fig.add_trace(go.Bar(
                name="Output",
                x=[m["name"] for m in models_data],
                y=[m["cost_per_1k_output_tokens"] * 1000 for m in models_data],
                marker_color="#2563eb",
            ))
            fig.update_layout(
                barmode="group",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font_color="white",
                yaxis_title="USD per 1M tokens",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Register New Model")
        with st.form("register_model"):
            col1, col2 = st.columns(2)
            name = col1.text_input("Model Name", placeholder="llama-3.1-8b-instant")
            provider = col2.selectbox("Provider", ["groq", "mistral", "ollama"])
            display_name = st.text_input("Display Name")
            description = st.text_area("Description (optional)")
            col1, col2, col3 = st.columns(3)
            context_window = col1.number_input("Context Window", value=8192, step=1024)
            input_cost = col2.number_input("Input Cost ($/1k tokens)", value=0.0, format="%.6f")
            output_cost = col3.number_input("Output Cost ($/1k tokens)", value=0.0, format="%.6f")
            submitted = st.form_submit_button("Register")
            if submitted and name and display_name:
                result = api_post("/api/v1/models/", {
                    "name": name,
                    "provider": provider,
                    "display_name": display_name,
                    "description": description or None,
                    "context_window": int(context_window),
                    "cost_per_1k_input_tokens": input_cost,
                    "cost_per_1k_output_tokens": output_cost,
                })
                if result:
                    st.success(f"Model '{name}' registered: `{result['id']}`")
                    st.cache_data.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Drift Monitor
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Drift Monitor":
    st.title("📉 Drift Monitor")

    summary = api_get("/api/v1/drift/summary")
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Monitored", summary["total_prompts"])
        col2.metric("Healthy", summary["healthy_prompts"])
        col3.metric(
            "Drifting",
            summary["drifting_prompts"],
            delta=None if summary["drifting_prompts"] == 0 else f"⚠️",
            delta_color="inverse",
        )
        col4.metric("Alerts (24h)", summary["alerts_last_24h"])

    st.markdown("---")

    # Recent logs
    st.subheader("Recent Drift Logs")
    logs = api_get("/api/v1/drift/logs?limit=50") or []
    if not logs:
        st.info("No drift logs yet. Trigger a check or wait for the scheduler.")
    else:
        df = pd.DataFrame([
            {
                "Prompt ID": l["prompt_id"][:8] + "...",
                "Version": l["prompt_version"],
                "Score": round(l["mean_composite_score"], 2),
                "Safety": round(l["mean_safety"], 2),
                "Alert": "🚨" if l["alert_fired"] else "✅",
                "Samples": l["sample_count"],
                "Date": l["created_at"][:16],
            }
            for l in logs
        ])
        st.dataframe(df, use_container_width=True)

        # Score timeline
        st.subheader("Score Timeline")
        if logs:
            timeline_df = pd.DataFrame([
                {
                    "date": l["created_at"],
                    "score": l["mean_composite_score"],
                    "prompt": l["prompt_id"][:8],
                }
                for l in logs
            ])
            fig = px.line(
                timeline_df,
                x="date",
                y="score",
                color="prompt",
                title="Mean Composite Score Over Time",
            )
            threshold = summary["alert_threshold"] if summary else 6.5
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
            fig.update_layout(
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

    if st.button("🔄 Trigger Manual Drift Check"):
        with st.spinner("Running drift check..."):
            result = api_post("/api/v1/drift/trigger", {})
        if result is not None:
            st.success(f"Drift check complete — {len(result) if isinstance(result, list) else '?'} snapshots created")
            st.cache_data.clear()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Cost Tracker
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Cost Tracker":
    st.title("💰 Cost Tracker")

    st.info(
        "Cost tracking aggregates token usage from experiment trials. "
        "Run some experiments to see data here."
    )

    experiments = api_get("/api/v1/experiments/") or []
    if experiments:
        st.subheader("Experiment Score vs Cost (estimated)")
        data = []
        for exp in experiments:
            if exp["trial_count"] > 0:
                data.append({
                    "name": exp["name"],
                    "trials": exp["trial_count"],
                    "mean_score_a": exp["mean_score_a"],
                    "mean_score_b": exp["mean_score_b"],
                    "status": exp["status"],
                })

        if data:
            df = pd.DataFrame(data)
            fig = px.scatter(
                df,
                x="trials",
                y="mean_score_a",
                size="trials",
                color="status",
                hover_name="name",
                title="Trials vs Score (Version A)",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

    models_data = api_get("/api/v1/models/") or []
    if models_data:
        st.subheader("Registered Model Cost Table")
        df = pd.DataFrame([
            {
                "Model": m["name"],
                "Provider": m["provider"],
                "$/1M Input": m["cost_per_1k_input_tokens"] * 1000,
                "$/1M Output": m["cost_per_1k_output_tokens"] * 1000,
                "Active": m["is_active"],
            }
            for m in models_data
        ])
        st.dataframe(df, use_container_width=True)
