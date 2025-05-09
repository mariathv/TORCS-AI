import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import custom modules
from data_preprocessing import load_and_preprocess_data

def load_trained_model(model_path):
    """Load a previously trained model for inference"""
    # Add .h5 extension if no extension is present
    if not model_path.endswith(('.keras', '.h5')):
        model_path += '.h5'
        
    if os.path.exists(model_path):
        try:
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            print(f"Model loaded from {model_path}")
            print(f"Model input shape: {model.input_shape}, Expected features: {model.input_shape[1]}")
            return model
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None
    else:
        print(f"No model found at {model_path}")
        return None

def load_scaler(scaler_path):
    """Load a scaler from the given path"""
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Scaler loaded from {scaler_path}")
            return scaler
        except Exception as e:
            print(f"Error loading scaler {scaler_path}: {e}")
            return None
    else:
        print(f"No scaler found at {scaler_path}")
        return None

def evaluate_model(model_path, scaler_path, data_path, output_dir='evaluation_results', 
                   sample_size=100, model_type=None):
    """
    Comprehensive evaluation of the trained model on test data
    
    Args:
        model_path: Path to the trained model file
        scaler_path: Path to the corresponding scaler file
        data_path: Path to the telemetry CSV data
        output_dir: Directory to save evaluation results
        sample_size: Number of random samples to visualize
        model_type: Type of the model (corner_handling, gear_selection, etc.)
    """
    # Create model type subfolder in output directory
    if model_type:
        output_subdir = os.path.join(output_dir, model_type)
    else:
        model_basename = os.path.basename(model_path).split('.')[0]
        output_subdir = os.path.join(output_dir, model_basename)
    
    os.makedirs(output_subdir, exist_ok=True)
    
    print("=" * 80)
    print(f"EVALUATING TORCS ML CONTROLLER MODEL: {model_path}")
    if model_type:
        print(f"MODEL TYPE: {model_type}")
    print("=" * 80)
    
    # Load the model
    model = load_trained_model(model_path)
    if not model:
        print("Error: Model could not be loaded.")
        return
        
    print(f"Model input shape: {model.input_shape}")
    print(f"Feature count: {model.input_shape[1]}")
    
    # Load the scaler
    scaler = load_scaler(scaler_path)
    if not scaler:
        print("Warning: Scaler could not be loaded. Using default preprocessing.")
    
    # Load and preprocess data
    print(f"Loading data from {data_path}")
    start_time = time.time()
    
    try:
        # Try using our scaler if provided
        if scaler:
            # Load raw data without scaling
            data = pd.read_csv(data_path)
            
            # Define input columns based on model type
            if model_type == 'corner_handling':
                input_cols = ['speedX', 'speedY', 'angle', 'trackPos', 'track_0', 'track_2', 'track_4', 
                             'track_8', 'track_10', 'track_14', 'track_16', 'track_18']
                output_cols = ['steer', 'brake', 'accel']
            elif model_type == 'gear_selection':
                input_cols = ['speedX', 'rpm', 'gear', 'track_0', 'track_10']
                output_cols = ['gear']
            elif model_type == 'tactical':
                input_cols = ['speedX', 'speedY', 'trackPos', 'track_0', 'track_5', 
                             'track_10', 'track_15', 'rpm', 'gear']
                output_cols = ['brake', 'accel']
            elif model_type == 'high_level':
                input_cols = ['speedX', 'speedY', 'angle', 'trackPos', 'track_0', 'track_5', 
                             'track_10', 'track_15', 'rpm', 'gear']
                output_cols = ['steer', 'brake', 'accel']
            elif model_type == 'low_level':
                input_cols = ['speedX', 'speedY', 'angle', 'trackPos', 'track_0', 'track_5', 
                             'track_10', 'track_15']
                output_cols = ['steer', 'brake', 'accel']
            else:
                # Default to use all available columns
                # Filter columns based on model input shape
                feature_count = model.input_shape[1]
                
                # Basic features always included
                input_cols = ['speedX', 'speedY', 'speedZ', 'angle', 'trackPos', 'rpm', 'gear']
                
                # Add track sensors based on remaining features
                track_sensors = feature_count - len(input_cols)
                for i in range(min(19, track_sensors)):
                    # Try both naming conventions
                    if f'track_{i}' in data.columns:
                        input_cols.append(f'track_{i}')
                    elif f'track[{i}]' in data.columns:
                        input_cols.append(f'track[{i}]')
                
                output_cols = ['steer', 'brake', 'accel']
            
            # Filter to columns that exist in the dataset
            input_cols = [col for col in input_cols if col in data.columns]
            output_cols = [col for col in output_cols if col in data.columns]
            
            print(f"Using input columns ({len(input_cols)} features): {input_cols}")
            print(f"Using output columns: {output_cols}")
            
            # Extract inputs and outputs
            X = data[input_cols].values
            y = data[output_cols].values
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Apply scaling using the provided scaler
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
        else:
            # Fall back to standard preprocessing
            X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
                data_path, return_scaler=True
            )
            
        print(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        # Try standard preprocessing as fallback
        try:
            X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
                data_path, return_scaler=True
            )
        except Exception as e2:
            print(f"Fatal error during data preprocessing: {e2}")
            return
    
    # Get predictions
    print("Generating predictions on test data...")
    y_pred = model.predict(X_test)
    
    # Handle case where model only predicts one value
    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        y_pred = y_pred.reshape(-1, 1)
        print(f"Reshaped predictions to {y_pred.shape}")
    
    # Determine output names based on model type or output shape
    if model_type == 'gear_selection':
        output_names = ['gear']
    elif y_test.shape[1] == 1:
        output_names = ['output']
    elif y_test.shape[1] == 3:
        output_names = ['steering', 'braking', 'acceleration']
    elif y_test.shape[1] == 2:
        output_names = ['braking', 'acceleration']
    else:
        output_names = [f'output_{i}' for i in range(y_test.shape[1])]
    
    # Calculate metrics
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
        if i < y_test.shape[1]:  # Safety check
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
        'Output': ['Overall'] + output_names[:y_test.shape[1]],
        'MSE': [metrics['overall']['MSE']] + [metrics[name]['MSE'] for name in output_names[:y_test.shape[1]]],
        'MAE': [metrics['overall']['MAE']] + [metrics[name]['MAE'] for name in output_names[:y_test.shape[1]]]
    })
    
    # Add R² for individual outputs (not applicable for overall)
    r2_values = [np.nan] + [metrics[name]['R²'] for name in output_names[:y_test.shape[1]]]
    metrics_df['R²'] = r2_values
    
    metrics_csv_path = os.path.join(output_subdir, 'metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")
    
    # Visualize predictions vs ground truth for random samples
    print(f"Generating visualization for {sample_size} random samples...")
    
    # Pick random indices from test set
    indices = np.random.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    
    # Create plots for each output
    plt.figure(figsize=(18, 12))
    
    for i, name in enumerate(output_names):
        if i >= y_test.shape[1]:  # Safety check
            continue
            
        plt.subplot(min(3, y_test.shape[1]), 1, i+1)
        
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
    scatter_plot_path = os.path.join(output_subdir, 'prediction_scatter.png')
    plt.savefig(scatter_plot_path)
    print(f"Scatter plot saved to {scatter_plot_path}")
    
    # Create error distribution plot
    plt.figure(figsize=(15, 5))
    
    for i, name in enumerate(output_names):
        if i >= y_test.shape[1]:  # Safety check
            continue
            
        plt.subplot(1, min(3, y_test.shape[1]), i+1)
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
    error_dist_path = os.path.join(output_subdir, 'error_distribution.png')
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
            # Check if we have defined input columns from earlier
            if 'input_cols' in locals() and isinstance(input_cols, list) and len(input_cols) == feature_importance.shape[0]:
                pass  # Use already defined input_cols
            else:
                # Basic features
                input_cols = ['speedX', 'speedY', 'speedZ', 'angle', 'trackPos', 'rpm', 'gear']
                
                # Add track sensor names based on model input shape
                track_sensor_count = model.input_shape[1] - 7  # Subtract basic features
                for i in range(track_sensor_count):
                    input_cols.append(f'track_{i}')
            
            # Ensure we have the right number of column names
            if len(input_cols) > feature_importance.shape[0]:
                input_cols = input_cols[:feature_importance.shape[0]]
            elif len(input_cols) < feature_importance.shape[0]:
                for i in range(len(input_cols), feature_importance.shape[0]):
                    input_cols.append(f'feature_{i}')
                
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
            
            feature_imp_path = os.path.join(output_subdir, 'feature_importance.png')
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
        if i >= y_test.shape[1]:  # Safety check
            continue
            
        plt.subplot(min(3, y_test.shape[1]), 1, i+1)
        
        actual = y_test[seq_range, i]
        predicted = y_pred[seq_range, i]
        
        plt.plot(actual, 'b-', label='Actual')
        plt.plot(predicted, 'r-', label='Predicted')
        plt.title(f'{name.capitalize()} - Sequential Predictions')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    sequence_path = os.path.join(output_subdir, 'prediction_sequence.png')
    plt.savefig(sequence_path)
    print(f"Prediction sequence plot saved to {sequence_path}")
    
    # Generate a comprehensive report in text format
    report_path = os.path.join(output_subdir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TORCS ML CONTROLLER EVALUATION REPORT\n")
        if model_type:
            f.write(f"MODEL TYPE: {model_type}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {model_path}\n")
        f.write(f"Scaler: {scaler_path}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"Feature count: {model.input_shape[1]}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Overall MSE: {overall_mse:.6f}\n")
        f.write(f"Overall MAE: {overall_mae:.6f}\n\n")
        
        f.write("PER-OUTPUT METRICS:\n")
        for name in output_names[:y_test.shape[1]]:
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
    
def evaluate_all_models(models_dir, scalers_dir, data_path, output_dir='all_models_evaluation'):
    """
    Evaluate all models in the given directory with their corresponding scalers
    
    Args:
        models_dir: Directory containing model files
        scalers_dir: Directory containing scaler files
        data_path: Path to the telemetry CSV data
        output_dir: Directory to save evaluation results
    """
    print("=" * 80)
    print(f"EVALUATING ALL MODELS IN: {models_dir}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all model files
    model_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(('.h5', '.keras')):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        return
    
    print(f"Found {len(model_files)} model files")
    
    # Evaluate each model
    summary_data = []
    
    for model_path in model_files:
        model_basename = os.path.basename(model_path).split('.')[0]
        
        # Determine model type from file name
        model_type = None
        if 'corner_handling' in model_basename:
            model_type = 'corner_handling'
        elif 'gear_selection' in model_basename:
            model_type = 'gear_selection'
        elif 'tactical' in model_basename:
            model_type = 'tactical'
        elif 'high_level' in model_basename:
            model_type = 'high_level'
        elif 'low_level' in model_basename:
            model_type = 'low_level'
        
        # Find corresponding scaler
        scaler_basename = model_basename
        if model_basename.endswith('_model'):
            scaler_basename = model_basename.replace('_model', '_feature_scaler')
        elif model_basename.endswith('_best'):
            scaler_basename = model_basename.replace('_best', '_feature_scaler')
            
        scaler_path = os.path.join(scalers_dir, f"{scaler_basename}.pkl")
        if not os.path.exists(scaler_path):
            # Try alternate naming patterns
            scaler_basename = model_basename.split('_')[0] + '_' + model_basename.split('_')[1] + '_feature_scaler'
            scaler_path = os.path.join(scalers_dir, f"{scaler_basename}.pkl")
            
        if not os.path.exists(scaler_path):
            print(f"Warning: No scaler found for {model_basename}")
            scaler_path = None
        
        print(f"\nEvaluating {model_basename} (Type: {model_type})")
        
        try:
            # Evaluate the model
            evaluate_model(
                model_path=model_path,
                scaler_path=scaler_path,
                data_path=data_path,
                output_dir=output_dir,
                model_type=model_type
            )
            
            # Load the generated metrics file
            if model_type:
                metrics_path = os.path.join(output_dir, model_type, 'metrics.csv')
            else:
                metrics_path = os.path.join(output_dir, model_basename, 'metrics.csv')
                
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path)
                
                # Get overall metrics
                overall_row = metrics_df[metrics_df['Output'] == 'Overall']
                overall_mse = overall_row['MSE'].values[0] if len(overall_row) > 0 else np.nan
                overall_mae = overall_row['MAE'].values[0] if len(overall_row) > 0 else np.nan
                
                # Get the average R² across all outputs (excluding overall)
                output_rows = metrics_df[metrics_df['Output'] != 'Overall']
                avg_r2 = output_rows['R²'].mean() if len(output_rows) > 0 else np.nan
                
                # Add to summary
                summary_data.append({
                    'Model': model_basename,
                    'Type': model_type or 'Unknown',
                    'Overall_MSE': overall_mse,
                    'Overall_MAE': overall_mae,
                    'Average_R²': avg_r2
                })
        except Exception as e:
            print(f"Error evaluating {model_basename}: {e}")
    
    # Create summary report
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by average R² (descending)
        summary_df = summary_df.sort_values('Average_R²', ascending=False)
        
        # Save to CSV
        summary_path = os.path.join(output_dir, 'model_comparison_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nModel comparison summary saved to {summary_path}")
        
        # Create comparison chart
        plt.figure(figsize=(12, 8))
        
        # Plot Average R² for each model
        plt.subplot(2, 1, 1)
        bars = plt.bar(summary_df['Model'], summary_df['Average_R²'])
        
        # Color bars by model type
        type_colors = {
            'corner_handling': 'blue',
            'gear_selection': 'green',
            'tactical': 'orange',
            'high_level': 'red',
            'low_level': 'purple',
            'Unknown': 'gray'
        }
        
        for i, bar in enumerate(bars):
            model_type = summary_df.iloc[i]['Type']
            color = type_colors.get(model_type, 'gray')
            bar.set_color(color)
        
        plt.title('Model Comparison - Average R²')
        plt.xlabel('Model')
        plt.ylabel('Average R²')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot Overall MSE for each model
        plt.subplot(2, 1, 2)
        bars = plt.bar(summary_df['Model'], summary_df['Overall_MSE'])
        
        # Color bars by model type (same as above)
        for i, bar in enumerate(bars):
            model_type = summary_df.iloc[i]['Type']
            color = type_colors.get(model_type, 'gray')
            bar.set_color(color)
        
        plt.title('Model Comparison - Overall MSE (lower is better)')
        plt.xlabel('Model')
        plt.ylabel('Overall MSE')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Add legend
        plt.figure(figsize=(8, 2))
        handles = [plt.Rectangle((0,0),1,1, color=color) for color in type_colors.values()]
        labels = list(type_colors.keys())
        plt.legend(handles, labels, loc='center', ncol=len(type_colors))
        plt.axis('off')
        
        plt.tight_layout()
        legend_path = os.path.join(output_dir, 'model_types_legend.png')
        plt.savefig(legend_path)
        
        # Save the comparison chart
        comparison_path = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(comparison_path)
        print(f"Model comparison chart saved to {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced evaluation of TORCS ML controller models")
    parser.add_argument("--model", type=str, help="Path to a specific model file")
    parser.add_argument("--scaler", type=str, help="Path to the corresponding scaler file")
    parser.add_argument("--models_dir", type=str, help="Directory containing multiple model files")
    parser.add_argument("--scalers_dir", type=str, help="Directory containing multiple scaler files")
    parser.add_argument("--data", type=str, default="../telemetry_logs/telemetry.csv", 
                        help="Path to telemetry data CSV file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--model_type", type=str, choices=['corner_handling', 'gear_selection', 
                                                          'tactical', 'high_level', 'low_level'],
                        help="Type of model to evaluate")
    parser.add_argument("--sample_size", type=int, default=100, 
                        help="Number of random samples to visualize")
    
    args = parser.parse_args()
    
    if args.model and os.path.exists(args.model):
        # Evaluate a single model
        print(f"Evaluating single model: {args.model}")
        evaluate_model(
            model_path=args.model,
            scaler_path=args.scaler,
            data_path=args.data,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            model_type=args.model_type
        )
    elif args.models_dir and os.path.exists(args.models_dir):
        # Evaluate all models in directory
        print(f"Evaluating all models in: {args.models_dir}")
        evaluate_all_models(
            models_dir=args.models_dir,
            scalers_dir=args.scalers_dir or args.models_dir,
            data_path=args.data,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()
        print("\nError: Please provide either a single model path or a directory containing models.")

if __name__ == "__main__":
    main()