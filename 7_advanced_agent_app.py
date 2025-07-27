import streamlit as st
import pandas as pd
import joblib
import time

# --- Initialize Session State ---
if 'contacted_customers' not in st.session_state:
    st.session_state.contacted_customers = []
if 'customer_to_review' not in st.session_state:
    st.session_state.customer_to_review = None

# --- High-Performance Agent Tools ---

def tool_preprocess_bulk(customers_df: pd.DataFrame):
    """
    OPTIMIZED: Preprocesses a full DataFrame of customers at once.
    """
    template = pd.read_csv('churn_preprocessed.csv')
    X_columns = template.drop('Churn', axis=1).columns
    
    # Use pd.get_dummies on the whole dataframe for efficiency
    processed_df = pd.get_dummies(customers_df, columns=[col for col in customers_df.columns if customers_df[col].dtype == 'object' and col != 'customerID'])

    # Align columns with the model's training data
    for col in X_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    return processed_df[X_columns]

def tool_find_high_risk_customer(df: pd.DataFrame, model, contacted_ids: list):
    """
    AUTONOMY (OPTIMIZED): Scans the customer base efficiently to find the highest-risk, uncontacted customer.
    """
    uncontacted_df = df[~df['customerID'].isin(contacted_ids)].copy()
    if uncontacted_df.empty:
        return None, None, None

    # Preprocess all uncontacted customers in a single, fast operation
    preprocessed_bulk_data = tool_preprocess_bulk(uncontacted_df)
    
    # Predict churn probability for all of them at once
    probabilities = model.predict_proba(preprocessed_bulk_data)[:, 1]
    uncontacted_df['churn_probability'] = probabilities
    
    # Find the highest risk customer
    highest_risk_customer_series = uncontacted_df.sort_values(by='churn_probability', ascending=False).iloc[0]
    
    # Get the corresponding preprocessed data for the top customer
    highest_risk_preprocessed_data = preprocessed_bulk_data.loc[highest_risk_customer_series.name]

    return highest_risk_customer_series, highest_risk_customer_series['churn_probability'], pd.DataFrame([highest_risk_preprocessed_data])


def tool_analyze_top_drivers(model, preprocessed_data: pd.DataFrame):
    """Analyzes the top 2-3 drivers of churn for a specific customer."""
    importances = pd.Series(model.feature_importances_, index=preprocessed_data.columns)
    customer_features = preprocessed_data.iloc[0]
    active_features = customer_features[customer_features > 0].index
    top_drivers = importances[active_features].sort_values(ascending=False).head(3).index.tolist()
    return top_drivers

def tool_generate_hyper_personalized_message(customer_data: pd.Series, drivers: list):
    """PERSONALIZATION: Crafts a nuanced message based on multiple churn drivers."""
    message = f"Hello {customer_data['customerID']},\n\n"
    message += "We're reaching out because we value your business. Our system indicated a potential churn risk, and we want to understand how we can improve your experience.\n\n"
    message += "Our analysis suggests a few factors might be at play:\n"
    for driver in drivers:
        message += f"- Concerns related to: **{driver.replace('_', ' ').title()}**\n"
    
    message += f"\nGiven your loyalty (tenure: {customer_data['tenure']} months), we want to be proactive. We can offer a personalized plan to address these points, including potential discounts on your **${customer_data['MonthlyCharges']:.2f}** monthly bill.\n\n"
    message += "Would you be open to a brief chat with a retention specialist?\n\nBest,\nThe Customer Loyalty Team"
    return message

# --- Streamlit UI ---
st.set_page_config(page_title="Autonomous Retention Agent", layout="wide")
st.title("🤖 Autonomous Customer Retention Agent")
st.markdown("This agent **autonomously** finds at-risk customers, **personalizes** an intervention, and waits for **interactive** approval before taking action.")

# --- Agent Control Panel ---
st.header("Agent Control Panel")
if st.button("🚀 Find and Propose Action for Next At-Risk Customer"):
    with st.spinner("Agent is scanning the customer base... (This is now much faster!)"):
        full_df = pd.read_csv('churn.csv')
        model = joblib.load('churn_model.joblib')
        
        customer, probability, preprocessed_customer_df = tool_find_high_risk_customer(full_df, model, st.session_state.contacted_customers)
        
        if customer is not None:
            drivers = tool_analyze_top_drivers(model, preprocessed_customer_df)
            message = tool_generate_hyper_personalized_message(customer, drivers)
            
            st.session_state.customer_to_review = {
                "data": customer,
                "probability": probability,
                "drivers": drivers,
                "message": message
            }
        else:
            st.success("No more at-risk customers to contact!")
            st.session_state.customer_to_review = None
    st.rerun()


# --- INTERACTION: Action Center ---
if st.session_state.customer_to_review:
    st.header("Action Center: Human Approval Required")
    review_data = st.session_state.customer_to_review
    customer_id = review_data['data']['customerID']

    st.subheader(f"Customer in Review: {customer_id}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{review_data['probability']:.0%}")
        st.write("**Top Churn Drivers:**")
        for driver in review_data['drivers']:
            st.markdown(f"- `{driver}`")

    with col2:
        st.write("**Proposed Personalized Message:**")
        st.text_area("", review_data['message'], height=250)

    approve_col, deny_col, _ = st.columns([1, 1, 5])
    if approve_col.button("✅ Approve & Send", key=f"approve_{customer_id}"):
        st.session_state.contacted_customers.append(customer_id)
        st.session_state.customer_to_review = None
        st.success(f"Action for {customer_id} approved. Message dispatched (simulated).")
        st.rerun()

    if deny_col.button("❌ Deny Action", key=f"deny_{customer_id}"):
        st.session_state.contacted_customers.append(customer_id) # Mark as handled
        st.session_state.customer_to_review = None
        st.warning(f"Action for {customer_id} denied. The customer will not be contacted again.")
        st.rerun()

# --- Activity Log ---
st.header("Activity Log")
if st.session_state.contacted_customers:
    st.write("Customers already reviewed or contacted in this session:")
    st.write(st.session_state.contacted_customers)
else:
    st.write("No customers have been contacted yet in this session.")