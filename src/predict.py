import joblib
import numpy as np
import pandas as pd

# Feature names (30 total)
FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Load trained model and scaler
scaler = joblib.load("models/scaler.pkl")
svm_model = joblib.load("models/svm_model.pkl")  # You can swap with knn_model.pkl or logreg_model.pkl


def predict_cancer(input_data):
    """
    Predict if the tumor is benign or malignant using a trained model.
    input_data should be a list/array with 30 features in the same order as training.
    """
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = svm_model.predict(input_scaled)[0]
    probability = svm_model.predict_proba(input_scaled)[0]

    return {
        "prediction": prediction,                 # 'B' or 'M'
        "probability_Benign": probability[0],
        "probability_Malignant": probability[1]
    }


def get_user_input():
    """Manual input for 30 features."""
    print("\n🔢 Please enter the following 30 features in order:")
    user_features = []
    for name in FEATURE_NAMES:
        val = float(input(f"{name}: "))
        user_features.append(val)
    return user_features


def get_sample_from_csv(index=0):
    """Get a sample row from the CSV for testing (default: first row)."""
    df = pd.read_csv("data/breast_cancer.csv", header=None)
    features = df.iloc[index, 2:32].values  # Columns 2-31 are features
    label = df.iloc[index, 1]               # Column 1 is diagnosis
    return features, label


if __name__ == "__main__":
    print("Choose input mode:")
    print("1 - Manual input of 30 features")
    print("2 - Use a sample from breast_cancer.csv")
    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        user_features = get_user_input()
        actual_label = None
    else:
        row_index = int(input("Enter row index from CSV (0 for first row): ") or 0)
        user_features, actual_label = get_sample_from_csv(row_index)
        print(f"\nLoaded sample row {row_index} from CSV. Actual label: {actual_label}")

    result = predict_cancer(user_features)

    print("\n✅ Prediction Result:")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability (Benign): {result['probability_Benign']:.4f}")
    print(f"Probability (Malignant): {result['probability_Malignant']:.4f}")

    if actual_label:
        print(f"Actual diagnosis: {actual_label}")
        print(f"Match: {result['prediction'] == actual_label}")
