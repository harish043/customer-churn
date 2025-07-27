import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(input_path, model_output_path):
    """
    Trains a churn prediction model and saves it.
    """
    df = pd.read_csv(input_path)

    # Separate features (X) and target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, model_output_path)
    print(f"\nModel saved to {model_output_path}")

if __name__ == "__main__":
    train_model('churn_preprocessed.csv', 'churn_model.joblib')