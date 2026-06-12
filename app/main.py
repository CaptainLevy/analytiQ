import streamlit as st
import pandas as pd
import json
import plotly.io as pio
import plotly.graph_objects as go

from app.agent import run_agent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AnalytiQ",
    page_icon="📊",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 AnalytiQ")
st.caption("Upload a dataset and ask questions in plain English.")

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None

# ── Sidebar: file upload + dataset preview ────────────────────────────────────
with st.sidebar:
    st.header("📁 Dataset")
    
    tab1, tab2 = st.tabs(["Upload", "Samples"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.chat_history = []
            st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    with tab2:
        st.markdown("Try a sample dataset:")
        if st.button("🛍️ Retail Sales"):
            st.session_state.df = pd.read_csv("data/retail_sales.csv")
            st.session_state.chat_history = []
            st.rerun()
        if st.button("👥 HR Attrition"):
            st.session_state.df = pd.read_csv("data/hr_attrition.csv")
            st.session_state.chat_history = []
            st.rerun()
        if st.button("📣 Marketing Campaign"):
            st.session_state.df = pd.read_csv("data/marketing_campaign.csv")
            st.session_state.chat_history = []
            st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df
        st.divider()
        st.subheader("Preview")
        st.dataframe(df.head(5), use_container_width=True)

        st.subheader("Column Types")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Missing": df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    
    st.subheader("💡 Example questions you can ask:")
    examples = [
        "Give me an overview of this dataset",
        "Are there any data quality issues?",
        "What is the correlation between numeric columns?",
        "Show me the distribution of sales",
        "Is there a significant difference in revenue across regions?",
        "Which category has the highest average profit?",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")

else:
    df = st.session_state.df

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "chart_json" in msg and msg["chart_json"]:
                fig = pio.from_json(msg["chart_json"])
                st.plotly_chart(fig, use_container_width=True)

    # Chat input
    user_input = st.chat_input("Ask anything about your data...")

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                result = run_agent(
                    user_query=user_input,
                    df=df,
                    chat_history=st.session_state.chat_history[:-1]
                )

            st.markdown(result["text"])

            if result.get("chart_json"):
                fig = pio.from_json(result["chart_json"])
                st.plotly_chart(fig, use_container_width=True)

            if result.get("tool_used"):
                st.caption(f"🔧 Tool used: `{result['tool_used']}`")

        # Save assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["text"],
            "chart_json": result.get("chart_json")
        })