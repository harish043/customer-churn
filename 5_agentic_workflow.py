# You may need to install langgraph: pip install langgraph
import pandas as pd
import joblib
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

# 1. Define the Agent's State
class AgentState(TypedDict):
    customer_data: dict
    preprocessed_data: pd.DataFrame
    churn_probability: float
    churn_risk: str
    top_churn_driver: str
    suggested_intervention: str

# 2. Define the Nodes (Workflow Steps)
def load_and_preprocess_data(state: AgentState):
    """Loads customer data and preprocesses it for the model."""
    print("---PREDICTING CHURN---")
    customer_data = state['customer_data']
    
    # Load preprocessed data for column structure
    preprocessed_df_template = pd.read_csv('churn_preprocessed.csv')
    X_columns = preprocessed_df_template.drop('Churn', axis=1).columns
    
    # Create a DataFrame for the new customer
    customer_df = pd.DataFrame([customer_data])
    
    # Simple preprocessing - align columns with the training data
    for col in X_columns:
        if col not in customer_df.columns:
            customer_df[col] = 0
    customer_df = customer_df[X_columns]
    
    return {"preprocessed_data": customer_df}

def predict_churn(state: AgentState):
    """Predicts churn using the trained model."""
    model = joblib.load('churn_model.joblib')
    preprocessed_data = state['preprocessed_data']
    
    # Predict probability
    churn_probability = model.predict_proba(preprocessed_data)[:, 1][0]
    churn_risk = "High" if churn_probability > 0.5 else "Low"
    
    print(f"Churn Probability: {churn_probability:.2f}")
    return {"churn_probability": churn_probability, "churn_risk": churn_risk}

def analyze_churn_reasons(state: AgentState):
    """Analyzes the reasons for churn risk."""
    print("---ANALYZING REASONS---")
    if state['churn_risk'] == "Low":
        return {"top_churn_driver": "N/A"}

    model = joblib.load('churn_model.joblib')
    feature_importances = pd.Series(model.feature_importances_, index=state['preprocessed_data'].columns)
    
    # Get the most important feature for this specific customer
    customer_data = state['preprocessed_data'].iloc[0]
    customer_features = customer_data[customer_data > 0].index
    
    top_driver = feature_importances[customer_features].idxmax()
    
    print(f"Top Churn Driver for this customer: {top_driver}")
    return {"top_churn_driver": top_driver}

def suggest_intervention(state: AgentState):
    """Suggests a personalized intervention."""
    print("---SUGGESTING INTERVENTION---")
    if state['churn_risk'] == "Low":
        intervention = "No immediate action needed. Continue standard customer engagement."
    else:
        driver = state['top_churn_driver']
        if 'tenure' in driver or 'Contract' in driver:
            intervention = "Proactive outreach recommended: Offer a contract extension with a loyalty discount."
        elif 'MonthlyCharges' in driver or 'TotalCharges' in driver:
            intervention = "Proactive outreach recommended: Offer a special discount on their monthly bill."
        else:
            intervention = "Proactive outreach recommended: Engage customer to understand their dissatisfaction."
            
    print(f"Suggested Intervention: {intervention}")
    return {"suggested_intervention": intervention}

# 3. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("load_and_preprocess", load_and_preprocess_data)
workflow.add_node("predict_churn", predict_churn)
workflow.add_node("analyze_reasons", analyze_churn_reasons)
workflow.add_node("suggest_intervention", suggest_intervention)

workflow.set_entry_point("load_and_preprocess")
workflow.add_edge("load_and_preprocess", "predict_churn")
workflow.add_edge("predict_churn", "analyze_reasons")
workflow.add_edge("analyze_reasons", "suggest_intervention")
workflow.add_edge("suggest_intervention", END)

app = workflow.compile()

# --- Run the Agent ---
if __name__ == '__main__':
    # Example of a high-risk customer (low tenure, high monthly charges)
    high_risk_customer = {
        'gender': 0, 'SeniorCitizen': 1, 'Partner': 0, 'Dependents': 0, 'tenure': 2,
        'PhoneService': 1, 'PaperlessBilling': 1, 'MonthlyCharges': 95.50, 'TotalCharges': 180.5,
        'MultipleLines_No': 0, 'MultipleLines_Yes': 1,
        'InternetService_Fiber optic': 1, 'InternetService_No': 0,
        'OnlineSecurity_No': 1, 'OnlineSecurity_Yes': 0,
        'OnlineBackup_No': 1, 'OnlineBackup_Yes': 0,
        'DeviceProtection_No': 1, 'DeviceProtection_Yes': 0,
        'TechSupport_No': 1, 'TechSupport_Yes': 0,
        'StreamingTV_Yes': 1, 'StreamingTV_No': 0,
        'StreamingMovies_Yes': 1, 'StreamingMovies_No': 0,
        'Contract_Month-to-month': 1, 'Contract_One year': 0, 'Contract_Two year': 0,
        'PaymentMethod_Electronic check': 1, 'PaymentMethod_Credit card (automatic)': 0, 'PaymentMethod_Mailed check': 0
    }

    inputs = {"customer_data": high_risk_customer}
    final_state = app.invoke(inputs)
    
    print("\n---AGENTIC WORKFLOW COMPLETE---")
    print(f"Final Analysis for Customer:")
    print(f"  - Churn Risk: {final_state['churn_risk']}")
    print(f"  - Churn Probability: {final_state['churn_probability']:.2f}")
    print(f"  - Top Churn Driver: {final_state['top_churn_driver']}")
    print(f"  - Suggested Intervention: {final_state['suggested_intervention']}")