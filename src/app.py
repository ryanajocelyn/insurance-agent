"""
Streamlit Web Interface for Multi-Agent Motor Claim Adjudication Assistant.
"""

import json
import streamlit as st
from src.config import config
from src.core.state import ClaimState
from src.core.graph import run_claim_adjudication

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Motor Claim Adjudication Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "current_claim_state" not in st.session_state:
        st.session_state.current_claim_state = None
    if "override_logs" not in st.session_state:
        st.session_state.override_logs = []


def render_sidebar():
    """Render sidebar configuration and engine parameter controls."""
    st.sidebar.title("⚙️ Engine Configuration")
    st.sidebar.info(
        f"**Framework**: LangGraph\n\n"
        f"**LLM**: {config.GEMINI_MODEL_NAME}\n\n"
        f"**Embeddings**: {config.EMBEDDING_MODEL_NAME}\n\n"
        f"**Vector Store**: ChromaDB\n\n"
        f"**RDBMS**: SQLite + JinjaSql"
    )

    st.sidebar.subheader("Operational Thresholds")
    st.sidebar.markdown(f"- **Adjustment Clamp**: +{config.COST_ADJUSTMENT_THRESHOLD * 100:.0f}%")
    st.sidebar.markdown(f"- **Fraud Escalation**: >+{config.FRAUD_ESCALATION_THRESHOLD * 100:.0f}%")
    st.sidebar.markdown(f"- **Velocity Lookback**: {config.CLAIM_VELOCITY_LOOKBACK_MONTHS} Months")
    st.sidebar.markdown(f"- **FIR Mandatory**: > ₹{config.FIR_REQUIRED_LOSS_THRESHOLD:,.0f}")


def render_packet_input_form():
    """Render Claim Packet Input Form."""
    st.subheader("📋 Claim Packet Submission")

    col1, col2 = st.columns(2)

    with col1:
        claim_id = st.text_input("Claim ID", value="CLM-2026-9041")
        policy_number = st.text_input("Policy Number", value="POL-OD-88192")
        make = st.selectbox("Vehicle Make", ["Maruti Suzuki", "Hyundai", "Honda", "Mahindra", "BMW", "Mercedes-Benz"])
        model = st.text_input("Vehicle Model", value="Swift VXi")
        vehicle_segment = st.selectbox("Vehicle Segment", ["hatchback", "sedan", "suv", "luxury"])
        vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0.1, max_value=15.0, value=2.5, step=0.5)
        cubic_capacity = st.number_input("Engine Capacity (CC)", min_value=600.0, max_value=5000.0, value=1197.0, step=100.0)

    with col2:
        narrative = st.text_area(
            "Accident Incident Narrative",
            value="Vehicle collided with a concrete pillar while reversing out of parking bay. Scratched rear bumper and dented tailgate.",
            height=130,
        )

        endorsements = st.multiselect(
            "Active Policy Endorsements / Riders",
            options=["ZERO_DEP", "ENGINE_PROTECT", "CONSUMABLES_COVER", "RETURN_TO_INVOICE", "ROADSIDE_ASSISTANCE"],
            default=["ZERO_DEP"],
        )

        st.markdown("**Estimate Repair Line Items**")
        line_item_1_name = st.text_input("Item 1 Name", value="Rear Bumper Assembly")
        line_item_1_cat = st.selectbox("Item 1 Category", ["plastic", "metal", "fiberglass", "glass", "labor"])
        line_item_1_cost = st.number_input("Item 1 Cost (₹)", min_value=0.0, value=6500.0, step=500.0)

        uploaded_files = st.file_uploader(
            "Upload Vehicle Damage Photographs",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
        )

    submit_button = st.button("🚀 Run Multi-Agent Adjudication Engine", type="primary", use_container_width=True)

    if submit_button:
        image_paths = []
        if uploaded_files:
            for file in uploaded_files:
                image_paths.append(file.name)

        initial_state: ClaimState = {
            "claim_id": claim_id,
            "policy_number": policy_number,
            "vehicle_details": {
                "make": make,
                "model": model,
                "segment": vehicle_segment,
                "age_years": vehicle_age,
                "cubic_capacity": cubic_capacity,
            },
            "incident_narrative": narrative,
            "uploaded_images": image_paths,
            "estimate_line_items": [
                {
                    "part_name": line_item_1_name,
                    "category": line_item_1_cat,
                    "claimed_cost": line_item_1_cost,
                }
            ],
            "customer_history": [],
            "held_policy_endorsements": endorsements,
        }

        with st.spinner("Executing LangGraph Parallel Multi-Agent Adjudication Workflow..."):
            final_state = run_claim_adjudication(initial_state)
            st.session_state.current_claim_state = final_state
            st.success("Adjudication Workflow Completed!")


def render_adjudication_results():
    """Render synthesized decision metrics, rationale, and trace visualizer."""
    state = st.session_state.current_claim_state
    if not state:
        return

    st.markdown("---")
    st.header("⚖️ Adjudication Decision Output")

    verdict = state.get("adjudication_verdict", "ESCALATE")

    if verdict == "APPROVE":
        st.success(f"### Verdict: APPROVED")
    elif verdict == "APPROVE_ADJUSTED":
        st.warning(f"### Verdict: APPROVED WITH ADJUSTMENT")
    elif verdict == "ESCALATE":
        st.error(f"### Verdict: ESCALATED FOR MANUAL REVIEW")
    else:
        st.error(f"### Verdict: REJECTED")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gross Claimed", f"₹{state.get('claimed_amount', 0.0):,.2f}")
    m2.metric("Approved Net Payout", f"₹{state.get('approved_amount', 0.0):,.2f}")
    m3.metric("Depreciation Deduction", f"₹{state.get('deductions_breakdown', {}).get('depreciation', 0.0):,.2f}")
    m4.metric("Compulsory Excess", f"₹{state.get('deductions_breakdown', {}).get('compulsory_deductible', 0.0):,.2f}")

    t1, t2, t3, t4 = st.tabs([
        "Audit Rationale",
        "Policy Citations & Warnings",
        "Agent Trace Findings",
        "📜 Execution Logs & Data Lineage"
    ])

    with t1:
        st.markdown("**Adjudication Rationale:**")
        st.write(state.get("adjudication_rationale", "No rationale generated."))

    with t2:
        st.markdown("**Mandatory Policy & Regulatory Citations:**")
        for cite in state.get("mandatory_citations", []):
            st.markdown(f"- 📜 `{cite}`")

        st.markdown("**Policy Warning Flags:**")
        warnings = state.get("policy_warning_flags", [])
        if warnings:
            for w in warnings:
                st.warning(f"⚠️ {w}")
        else:
            st.info("No policy warnings emitted.")

    with t3:
        st.json(
            {
                "vision_findings": state.get("vision_findings", {}),
                "cross_modal_consistency": state.get("cross_modal_consistency", True),
                "missing_evidence_flags": state.get("missing_evidence_flags", []),
                "cost_variance_flags": state.get("cost_variance_flags", []),
                "frequency_risk_score": state.get("frequency_risk_score", 0.0),
                "investigation_triggers": state.get("investigation_triggers", []),
            }
        )

    with t4:
        st.subheader("📋 Step-by-Step Agent Execution & Data Origin Provenance")
        st.markdown(
            "Below is the minimal execution log trace showing the order of steps executed by each agent node "
            "and the exact external data sources, database collections, and statutory schedules queried."
        )

        logs = state.get("execution_logs", [])
        if not logs:
            st.info("No execution logs found in current state.")
        else:
            for idx, log in enumerate(logs, 1):
                status_icon = "🟢" if log.get("status") == "SUCCESS" else ("🟡" if log.get("status") == "WARNING" else "🔴")
                agent_name = log.get("agent", "Agent")
                step_title = log.get("step", "Execution Step")
                timestamp = log.get("timestamp", "")
                
                expander_label = f"Step {idx}: {status_icon} [{agent_name}] {step_title} — {timestamp}"
                
                with st.expander(expander_label, expanded=(idx == len(logs))):
                    st.markdown(f"**Agent**: `{agent_name}`")
                    st.markdown(f"**Step**: `{step_title}`")
                    st.markdown(f"**Execution Status**: `{log.get('status', 'SUCCESS')}`")
                    st.markdown(f"**Step Rationale & Summary**:\n> {log.get('summary', '')}")
                    
                    st.markdown("---")
                    st.markdown("##### 📍 Data Sources & Information Provenance:")
                    sources = log.get("data_sources", [])
                    if sources:
                        for src in sources:
                            st.markdown(f"- 🗂️ `{src}`")
                    else:
                        st.caption("No specific external data sources recorded for this step.")


    st.markdown("---")
    st.subheader("👤 Human-in-the-Loop (HITL) Adjuster Override & Authorization")

    with st.expander("Interactive Adjuster Payout Override Panel", expanded=False):
        human_payout = st.number_input(
            "Modify Approved Payout Amount (₹)",
            min_value=0.0,
            value=float(state.get("approved_amount", 0.0)),
            step=500.0,
        )
        override_notes = st.text_area("Human Adjuster Sign-off Notes", value="Approved after manual photo verification.")
        confirm_override = st.button("Submit Human Adjuster Authorization")

        if confirm_override:
            log_entry = {
                "claim_id": state.get("claim_id"),
                "agent_payout": state.get("approved_amount"),
                "adjusted_payout": human_payout,
                "notes": override_notes,
            }
            st.session_state.override_logs.append(log_entry)
            st.success(f"Human Adjuster Override Recorded! Approved Payout updated to ₹{human_payout:,.2f}")


def main():
    init_session_state()
    render_sidebar()
    st.title("🛡️ Motor Insurance Claim Adjudication Assistant")
    st.markdown("*Multi-Agent LangGraph Engine powered by Google Gemini 1.5 Pro & ChromaDB RAG*")
    st.markdown("---")

    render_packet_input_form()
    render_adjudication_results()


if __name__ == "__main__":
    main()
