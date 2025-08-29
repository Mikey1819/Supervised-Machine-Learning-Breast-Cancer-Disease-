import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a single model and print metrics"""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="M")
    rec = recall_score(y_test, y_pred, pos_label="M")
    f1 = f1_score(y_test, y_pred, pos_label="M")

    print(f"\n📊 Results for {model_name}:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

def main():
    # Load dataset (after converting wdbc.data to CSV)
    df = pd.read_csv("data/breast_cancer.csv")

    # Drop id column (not useful for prediction)
    df = df.drop("id", axis=1)

    # Features (X) and target (y)
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    # Train/test split
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (important for SVM, KNN, etc.)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Load saved models (trained in train_models.py)
    svm_model = joblib.load("models/svm_model.pkl")
    knn_model = joblib.load("models/knn_model.pkl")
    logreg_model = joblib.load("models/logreg_model.pkl")

    # Evaluate each model
    results = []
    results.append(evaluate_model(svm_model, X_test, y_test, "SVM"))
    results.append(evaluate_model(knn_model, X_test, y_test, "KNN"))
    results.append(evaluate_model(logreg_model, X_test, y_test, "Logistic Regression"))

    # Convert results to DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    print("\n📈 Model Comparison:")
    print(results_df)

if __name__ == "__main__":
    main()