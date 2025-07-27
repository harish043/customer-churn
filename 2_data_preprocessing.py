import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def preprocess_data(input_path, output_path):
    """
    Loads, preprocesses, and saves the churn dataset.
    """
    df = pd.read_csv(input_path)

    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill any resulting NaN values with the median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Drop customerID
    df = df.drop('customerID', axis=1)

    # Encode binary categorical features
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # One-hot encode multi-class categorical features
    multi_class_cols = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)

    # Save the preprocessed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    print("\nFirst 5 rows of preprocessed data:")
    print(df.head())

if __name__ == "__main__":
    preprocess_data('churn.csv', 'churn_preprocessed.csv')