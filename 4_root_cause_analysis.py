import pandas as pd
import joblib

def analyze_churn_drivers(model_path, data_path):
    """
    Loads a trained model and identifies the top churn drivers.
    """
    # Load the trained model
    model = joblib.load(model_path)

    # Load the preprocessed data to get feature names
    df = pd.read_csv(data_path)
    X = df.drop('Churn', axis=1)

    # Get feature importances
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 10 Churn Drivers:")
    print(feature_importances.head(10))

if __name__ == "__main__":
    analyze_churn_drivers('churn_model.joblib', 'churn_preprocessed.csv')