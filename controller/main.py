from data_preprocessing import load_and_preprocess_data
from model_training import build_and_train_model

def main():
    #---- Load and preprocess data ----
    X_train, X_test, y_train, y_test = load_and_preprocess_data('../telemetry_logs/telemetry.csv')
    
    # ---- Build and train the model ----
    model = build_and_train_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
