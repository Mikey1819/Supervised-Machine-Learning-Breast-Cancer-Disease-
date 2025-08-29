import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(filepath="data/breast_cancer.csv"):
    """
    Load the dataset from CSV and return a DataFrame.
    """
    df = pd.read_csv(filepath)

    # Drop 'id' column (not useful for prediction)
    if "id" in df.columns:
        df = df.drop("id", axis=1)

    return df

def prepare_features_and_labels(df):
    """
    Separate features (X) and target (y).
    """
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split into train/test sets and apply scaling.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features (important for SVM, KNN, Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def predict_cancer(input_data):
    # Convert input to DataFrame with feature names
    input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    input_scaled = scaler.transform(input_df)
    prediction = svm_model.predict(input_scaled)[0]
    probability = svm_model.predict_proba(input_scaled)[0]
    result = {
        "prediction": prediction,
        "probability_Benign": probability[0],
        "probability_Malignant": probability[1]
    }
    return result

if __name__ == "__main__":
    # Example usage (for testing this file directly)
    df = load_dataset("data/breast_cancer.csv")
    X, y = prepare_features_and_labels(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    print("✅ Dataset loaded and preprocessed successfully.")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")