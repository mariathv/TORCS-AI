import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_preprocessing import load_and_preprocess_data

def evaluate_model_performance(model_path='controller/model.keras', data_path='../telemetry_logs/telemetry.csv'):
    """
    Evaluate a trained model to determine if it's optimal.
    
    Parameters:
    - model_path: Path to the trained model file
    - data_path: Path to the telemetry data for evaluation
    
    Returns:
    - Dictionary with evaluation metrics and optimization status
    """
    # Ensure model path has the correct extension
    if not model_path.endswith(('.keras', '.h5')):
        model_path += '.keras'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Load and preprocess data
    print(f"Loading data from {data_path}")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Evaluate on test data
    print("Evaluating model on test data...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Calculate additional metrics
    mse = np.mean(np.square(y_test - y_pred))
    rmse = np.sqrt(mse)
    
    # Calculate per-output metrics (steering, braking, acceleration)
    output_names = ['steering', 'braking', 'acceleration']
    output_metrics = []
    
    for i in range(y_test.shape[1]):
        output_mae = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
        output_mse = np.mean(np.square(y_test[:, i] - y_pred[:, i]))
        output_metrics.append({
            'name': output_names[i],
            'mae': output_mae,
            'mse': output_mse
        })
        print(f"{output_names[i].capitalize()} - MAE: {output_mae:.4f}, MSE: {output_mse:.4f}")
    
    # Determine if model is optimal based on thresholds
    # These thresholds can be adjusted based on your specific requirements
    is_optimal = test_mae < 0.1  # Example threshold
    optimization_status = "OPTIMAL" if is_optimal else "NEEDS IMPROVEMENT"
    
    print(f"\nModel Optimization Status: {optimization_status}")
    if not is_optimal:
        print("Suggestions for improvement:")
        print("1. Try increasing the number of training epochs")
        print("2. Experiment with different network architectures")
        print("3. Add more training data or improve data quality")
        print("4. Adjust learning rate or try different optimizers")
    
    # Visualize predictions vs actual values
    plt.figure(figsize=(15, 5))
    
    for i, name in enumerate(output_names):
        plt.subplot(1, 3, i+1)
        plt.scatter(y_test[:100, i], y_pred[:100, i], alpha=0.5)
        plt.plot([-1, 1], [-1, 1], 'r--')  # Perfect prediction line
        plt.title(f'{name.capitalize()} Predictions')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
    
    plt.tight_layout()
    plot_path = f"{os.path.splitext(model_path)[0]}_evaluation.png"
    plt.savefig(plot_path)
    print(f"Evaluation plot saved to {plot_path}")
    
    return {
        'test_loss': test_loss,
        'test_mae': test_mae,
        'rmse': rmse,
        'output_metrics': output_metrics,
        'is_optimal': is_optimal
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate TORCS ML model optimality')
    parser.add_argument('--model', type=str, default='controller/model.keras', 
                        help='Path to the trained model file')
    parser.add_argument('--data', type=str, default='../telemetry_logs/telemetry.csv', 
                        help='Path to the telemetry data for evaluation')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model_performance(args.model, args.data) 