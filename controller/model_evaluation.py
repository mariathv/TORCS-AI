import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import custom modules
from data_preprocessing import load_and_preprocess_data
from model_training import load_trained_model

def evaluate_model(model_path, data_path, output_dir='evaluation_results', sample_size=100):
    """
    Comprehensive evaluation of the trained model on test data
    
    Args:
        model_path: Path to the trained model file
        data_path: Path to the telemetry CSV data
        output_dir: Directory to save evaluation results
        sample_size: Number of random samples to visualize
    """
    print("=" * 80)
    print(f"EVALUATING TORCS ML CONTROLLER MODEL: {model_path}")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = load_trained_model(model_path)
    if not model:
        print("Error: Model could not be loaded.")
        return
        
    print(f"Model input shape: {model.input_shape}")
    print(f"Feature count: {model.input_shape[1]}")
    
    # Load and preprocess data
    print(f"Loading data from {data_path}")
    start_time = time.time()
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        data_path, return_scaler=True
    )
    print(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")
    
    # Get predictions
    print("Generating predictions on test data...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    output_names = ['steering', 'braking', 'acceleration']
    metrics = {}
    
    print("\nPERFORMANCE METRICS:")
    print("-" * 50)
    
    # Calculate overall metrics
    overall_mse = mean_squared_error(y_test, y_pred)
    overall_mae = mean_absolute_error(y_test, y_pred)
    metrics['overall'] = {'MSE': overall_mse, 'MAE': overall_mae}
    
    print(f"Overall MSE: {overall_mse:.6f}")
    print(f"Overall MAE: {overall_mae:.6f}")
    
    # Calculate per-output metrics
    print("\nPER-OUTPUT METRICS:")
    for i, name in enumerate(output_names):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        metrics[name] = {'MSE': mse, 'MAE': mae, 'R²': r2}
        
        print(f"{name.capitalize()}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²:  {r2:.6f}")
        
        # Calculate distribution of errors
        errors = np.abs(y_test[:, i] - y_pred[:, i])
        error_percentiles = {
            '50%': np.percentile(errors, 50),
            '90%': np.percentile(errors, 90),
            '95%': np.percentile(errors, 95),
            '99%': np.percentile(errors, 99),
            'max': np.max(errors)
        }
        
        print(f"  Error Distribution:")
        for p, v in error_percentiles.items():
            print(f"    {p}: {v:.6f}")
        print()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Output': ['Overall'] + output_names,
        'MSE': [metrics['overall']['MSE']] + [metrics[name]['MSE'] for name in output_names],
        'MAE': [metrics['overall']['MAE']] + [metrics[name]['MAE'] for name in output_names]
    })
    
    # Add R² for individual outputs (not applicable for overall)
    r2_values = [np.nan] + [metrics[name]['R²'] for name in output_names]
    metrics_df['R²'] = r2_values
    
    metrics_csv_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")
    
    # Visualize predictions vs ground truth for random samples
    print(f"Generating visualization for {sample_size} random samples...")
    
    # Pick random indices from test set
    indices = np.random.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    
    # Create plots for each output
    plt.figure(figsize=(18, 12))
    
    for i, name in enumerate(output_names):
        plt.subplot(3, 1, i+1)
        
        # Sort by ground truth for better visualization
        sorted_indices = np.argsort(y_test[indices, i])
        sorted_actual = y_test[indices, i][sorted_indices]
        sorted_predicted = y_pred[indices, i][sorted_indices]
        
        plt.scatter(range(len(sorted_indices)), sorted_actual, color='blue', label='Actual')
        plt.scatter(range(len(sorted_indices)), sorted_predicted, color='red', alpha=0.5, label='Predicted')
        
        plt.title(f'{name.capitalize()} - Actual vs Predicted')
        plt.ylabel('Value')
        plt.legend()
        
        # Add error bands
        errors = np.abs(sorted_actual - sorted_predicted)
        plt.fill_between(
            range(len(sorted_indices)),
            sorted_actual - errors,
            sorted_actual + errors,
            color='gray',
            alpha=0.2,
            label='Error'
        )
    
    plt.tight_layout()
    scatter_plot_path = os.path.join(output_dir, 'prediction_scatter.png')
    plt.savefig(scatter_plot_path)
    print(f"Scatter plot saved to {scatter_plot_path}")
    
    # Create error distribution plot
    plt.figure(figsize=(15, 5))
    
    for i, name in enumerate(output_names):
        plt.subplot(1, 3, i+1)
        errors = np.abs(y_test[:, i] - y_pred[:, i])
        
        plt.hist(errors, bins=50, alpha=0.7)
        plt.title(f'{name.capitalize()} - Error Distribution')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        
        # Add vertical lines for percentiles
        percentiles = [50, 90, 95]
        colors = ['g', 'orange', 'r']
        
        for p, c in zip(percentiles, colors):
            value = np.percentile(errors, p)
            plt.axvline(x=value, color=c, linestyle='--', 
                       label=f'{p}th percentile: {value:.4f}')
        
        plt.legend()
    
    plt.tight_layout()
    error_dist_path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(error_dist_path)
    print(f"Error distribution plot saved to {error_dist_path}")
    
    # Create correlation heatmap for the most important input features
    if hasattr(model, 'layers') and len(model.layers) > 0:
        plt.figure(figsize=(12, 10))
        
        # Get feature weights from first layer
        try:
            first_layer_weights = model.layers[0].get_weights()[0]
            # Sum absolute weights for each input feature
            feature_importance = np.sum(np.abs(first_layer_weights), axis=1)
            
            # Get original feature names (before scaling)
            input_cols = []
            # Basic features
            input_cols.extend(['speedX', 'speedY', 'speedZ', 'angle', 'trackPos', 'rpm', 'gear'])
            
            # Add track sensor names based on model input shape
            track_sensor_count = model.input_shape[1] - 7  # Subtract basic features
            for i in range(track_sensor_count):
                input_cols.append(f'track_{i}')
                
            # Sort features by importance
            sorted_idx = np.argsort(-feature_importance)
            top_features = sorted_idx[:min(15, len(sorted_idx))]
            
            # Create bar chart of feature importances
            plt.bar(
                [input_cols[i] for i in top_features],
                [feature_importance[i] for i in top_features]
            )
            plt.title('Feature Importance (Based on First Layer Weights)')
            plt.xlabel('Feature')
            plt.ylabel('Absolute Weight Sum')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            feature_imp_path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(feature_imp_path)
            print(f"Feature importance plot saved to {feature_imp_path}")
        except Exception as e:
            print(f"Could not generate feature importance plot: {e}")
    
    # Generate prediction sequence visualization (sequential frames)
    plt.figure(figsize=(15, 10))
    
    # Take a continuous sequence for this visualization
    seq_length = min(200, len(X_test))
    seq_start = np.random.randint(0, len(X_test) - seq_length)
    seq_range = range(seq_start, seq_start + seq_length)
    
    for i, name in enumerate(output_names):
        plt.subplot(3, 1, i+1)
        
        actual = y_test[seq_range, i]
        predicted = y_pred[seq_range, i]
        
        plt.plot(actual, 'b-', label='Actual')
        plt.plot(predicted, 'r-', label='Predicted')
        plt.title(f'{name.capitalize()} - Sequential Predictions')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    sequence_path = os.path.join(output_dir, 'prediction_sequence.png')
    plt.savefig(sequence_path)
    print(f"Prediction sequence plot saved to {sequence_path}")
    
    # Generate a comprehensive report in text format
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TORCS ML CONTROLLER EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"Feature count: {model.input_shape[1]}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Overall MSE: {overall_mse:.6f}\n")
        f.write(f"Overall MAE: {overall_mae:.6f}\n\n")
        
        f.write("PER-OUTPUT METRICS:\n")
        for name in output_names:
            f.write(f"{name.capitalize()}:\n")
            for metric, value in metrics[name].items():
                f.write(f"  {metric}: {value:.6f}\n")
            f.write("\n")
            
        f.write("\nEVALUATION SUMMARY:\n")
        f.write("-" * 50 + "\n")
        
        # Add some interpretation of the results
        for name, metric_dict in metrics.items():
            if name == 'overall':
                continue
                
            r2_value = metric_dict.get('R²', 0)
            if r2_value > 0.9:
                quality = "Excellent"
            elif r2_value > 0.8:
                quality = "Very good"
            elif r2_value > 0.7:
                quality = "Good"
            elif r2_value > 0.6:
                quality = "Moderate"
            elif r2_value > 0.5:
                quality = "Fair"
            else:
                quality = "Poor"
                
            f.write(f"{name.capitalize()} prediction quality: {quality} (R² = {r2_value:.4f})\n")
            
        f.write("\nGenerated visualizations:\n")
        f.write(f"- Scatter plot: {os.path.basename(scatter_plot_path)}\n")
        f.write(f"- Error distribution: {os.path.basename(error_dist_path)}\n")
        f.write(f"- Prediction sequence: {os.path.basename(sequence_path)}\n")
        f.write(f"- Feature importance: {os.path.basename(feature_imp_path)}\n")
        
    print(f"Evaluation report saved to {report_path}")
    print("\nEvaluation complete!")
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate TORCS ML controller model")
    parser.add_argument("--model", type=str, default="model", help="Path to the model file")
    parser.add_argument("--data", type=str, default="../telemetry_logs/telemetry.csv", help="Path to telemetry data")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory for evaluation results")
    parser.add_argument("--samples", type=int, default=100, help="Number of random samples to visualize")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        sample_size=args.samples
    )

if __name__ == "__main__":
    main()