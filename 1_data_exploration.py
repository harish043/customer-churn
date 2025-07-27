import pandas as pd

def explore_data(file_path):
    """
    Reads a CSV file and performs basic data exploration.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Print the first 5 rows
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\n" + "="*50 + "\n")

        # Print the column names and data types
        print("Column names and data types:")
        print(df.info())
        print("\n" + "="*50 + "\n")

        # Print summary statistics
        print("Summary statistics:")
        print(df.describe())
        print("\n" + "="*50 + "\n")

        # Print the number of missing values in each column
        print("Missing values per column:")
        print(df.isnull().sum())
        print("\n" + "="*50 + "\n")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    explore_data('churn.csv')