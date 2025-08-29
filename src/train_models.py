import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from preprocessing import load_dataset, prepare_features_and_labels, split_and_scale

def train_and_save_models():
    # Load and preprocess dataset
    df = load_dataset("data/breast_cancer.csv")
    X, y = prepare_features_and_labels(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Initialize models
    svm_model = SVC(kernel="linear", probability=True, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    logreg_model = LogisticRegression(max_iter=500, random_state=42)

    # Train models
    print("🚀 Training models...")
    svm_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    logreg_model.fit(X_train, y_train)
    print("✅ Training complete.")

    # Save models and scaler
    joblib.dump(svm_model, "models/svm_model.pkl")
    joblib.dump(knn_model, "models/knn_model.pkl")
    joblib.dump(logreg_model, "models/logreg_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("💾 Models and scaler saved in /models folder.")

if __name__ == "__main__":
    train_and_save_models()