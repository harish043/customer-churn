import streamlit as st
import pandas as pd
import joblib
import time
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

# --- Enhanced Agent State ---
class AgentState(TypedDict):
    customer_data: dict
    preprocessed_data: pd.DataFrame
    churn_probability: float
    churn_risk: str
    top_churn_driver: str
    intervention_plan: str
    personalized_message: str
    message_status: str

# --- Expanded Agent Tools ---
def tool_preprocess_data(state: AgentState):
    st.info("Tool: `preprocess_data`", icon="⚙️")
    with st.spinner("Preprocessing customer data..."):
        time.sleep(1)
        # (Code for preprocessing is the same as before)
        template = pd.read_csv('churn_preprocessed.csv')
        X_columns = template.drop('Churn', axis=1).columns
        df = pd.DataFrame([state['customer_data']])
        for col in X_columns:
            if col not in df.columns: df[col] = 0
        df = df[X_columns]
        st.write("✅ Preprocessing Complete.")
    return {"preprocessed_data": df}

def tool_predict_churn(state: AgentState):
    st.info("Tool: `predict_churn`", icon="🔮")
    with st.spinner("Running prediction model..."):
        time.sleep(1)
        model = joblib.load('churn_model.joblib')
        prob = model.predict_proba(state['preprocessed_data'])[:, 1][0]
        risk = "High" if prob > 0.5 else "Low"
        st.write(f"✅ Prediction Complete. Churn Risk: **{risk}**")
    return {"churn_probability": prob, "churn_risk": risk}

def tool_analyze_drivers(state: AgentState):
    st.info("Tool: `analyze_churn_drivers`", icon="📊")
    with st.spinner("Analyzing root causes..."):
        time.sleep(1)
        model = joblib.load('churn_model.joblib')
        importances = pd.Series(model.feature_importances_, index=state['preprocessed_data'].columns)
        customer_features = state['preprocessed_data'].iloc[0]
        active_features = customer_features[customer_features > 0].index
        top_driver = importances[active_features].idxmax()
        st.write("✅ Analysis Complete.")
    return {"top_churn_driver": top_driver}

def tool_plan_intervention(state: AgentState):
    st.info("Tool: `plan_intervention`", icon="💡")
    with st.spinner("Formulating intervention plan..."):
        time.sleep(1)
        driver = state.get('top_churn_driver', 'N/A')
        if 'tenure' in driver or 'Contract' in driver:
            plan = "Offer a contract extension with a loyalty discount."
        elif 'MonthlyCharges' in driver or 'TotalCharges' in driver:
            plan = "Offer a special discount on their monthly bill."
        else:
            plan = "Engage customer to understand their dissatisfaction."
        st.write("✅ Intervention Plan Created.")
    return {"intervention_plan": plan}

def tool_generate_personalized_message(state: AgentState):
    """NEW TOOL: Generates a personalized message based on the plan."""
    st.info("Tool: `generate_personalized_message`", icon="✉️")
    with st.spinner("Crafting personalized message..."):
        time.sleep(1.5)
        plan = state['intervention_plan']
        message = f"Hello,\n\nWe noticed you might be considering a change. We value you as a customer and want to help. Based on our analysis, we'd like to proactively offer the following: {plan}\n\nPlease let us know if you'd like to discuss this. We're here for you.\n\nBest,\nThe Customer Retention Team"
        st.write("✅ Message Generated.")
    return {"personalized_message": message}

def tool_dispatch_message(state: AgentState):
    """NEW TOOL: Simulates dispatching the message."""
    st.info("Tool: `dispatch_message`", icon="🚀")
    with st.spinner("Dispatching message via simulated API..."):
        time.sleep(1)
        st.write("✅ Message Dispatched.")
    return {"message_status": "Sent"}

# --- Conditional Logic for the Graph ---
def should_take_action(state: AgentState):
    st.info("Decision: `should_take_action`", icon="🤔")
    with st.spinner("Evaluating churn risk for action..."):
        time.sleep(1)
        if state['churn_risk'] == "High":
            st.write("✅ Verdict: High risk. Proceeding to proactive intervention.")
            return "analyze_drivers"
        else:
            st.write("✅ Verdict: Low risk. No action needed.")
            return END

# --- Build the Proactive Agent Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("preprocess", tool_preprocess_data)
workflow.add_node("predict", tool_predict_churn)
workflow.add_node("analyze_drivers", tool_analyze_drivers)
workflow.add_node("plan_intervention", tool_plan_intervention)
workflow.add_node("generate_message", tool_generate_personalized_message)
workflow.add_node("dispatch_message", tool_dispatch_message)

workflow.set_entry_point("preprocess")
workflow.add_edge("preprocess", "predict")
workflow.add_conditional_edges("predict", should_take_action, {"analyze_drivers": "analyze_drivers", "__end__": END})
workflow.add_edge("analyze_drivers", "plan_intervention")
workflow.add_edge("plan_intervention", "generate_message")
workflow.add_edge("generate_message", "dispatch_message")
workflow.add_edge("dispatch_message", END)
app = workflow.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="Proactive Churn Agent", layout="wide")
st.title("🤖 Proactive Customer Churn Agent")
st.markdown("This agent analyzes churn risk and **takes action** by sending personalized messages.")

with st.form("customer_data_form"):
    st.header("Enter Customer Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.number_input("Tenure (months)", 0, 100, 1)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 95.0)
    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col3:
        paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])

    submitted = st.form_submit_button("Run Proactive Agent")

if submitted:
    st.header("Agent Execution Log")
    customer_input = {
        'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': monthly_charges * tenure,
        'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0
    }
    
    final_state = app.invoke({"customer_data": customer_input})

    st.header("Final Result")
    if final_state.get("message_status") == "Sent":
        st.success("Proactive intervention was successful.")
        st.text_area("Message Sent to Customer", final_state['personalized_message'], height=200)
        st.metric(label="Message Status", value=final_state['message_status'])
    else:
        st.success("Customer is low-risk. No action was needed.")