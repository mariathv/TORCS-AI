import os
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_processor import TelemetryDataProcessor
from models import BaseModel

def load_model(model_type, use_best=True):
    """Load a trained model from disk
    
    Args:
        model_type (str): The type of model to load
        use_best (bool): Whether to load the best checkpoint or the final model
        
    Returns:
        The loaded model
    """
    if use_best:
        model_path = f'models/{model_type}_best.keras'
    else:
        model_path = f'models/{model_type}_model.keras'
        
    print(f"Loading model from {model_path}")
    return BaseModel.load(model_path)

def load_scalers(model_type):
    """Load feature and target scalers for a model
    
    Args:
        model_type (str): The type of model
        
    Returns:
        tuple: (feature_scaler, target_scaler)
    """
    feature_scaler_path = f'model_scalers/{model_type}_feature_scaler.pkl'
    target_scaler_path = f'model_scalers/{model_type}_target_scaler.pkl'
    
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    
    return feature_scaler, target_scaler

def evaluate_model(model, X_val, y_val, target_scaler=None):
    """Evaluate a model and return various metrics
    
    Args:
        model (tf.keras.Model): The model to evaluate
        X_val (np.ndarray): Validation input data
        y_val (np.ndarray): Validation target data
        target_scaler (object, optional): Scaler for inverting target normalization
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_val)
    
    # Inverse transform if scaler is provided
    if target_scaler is not None:
        y_val_orig = target_scaler.inverse_transform(y_val)
        y_pred_orig = target_scaler.inverse_transform(y_pred)
        
        # Calculate metrics on original scale
        mse_orig = mean_squared_error(y_val_orig, y_pred_orig, multioutput='raw_values')
        mae_orig = mean_absolute_error(y_val_orig, y_pred_orig, multioutput='raw_values')
        r2_orig = r2_score(y_val_orig, y_pred_orig, multioutput='raw_values')
    else:
        y_val_orig = y_val
        y_pred_orig = y_pred
        
    # Calculate metrics on normalized scale
    mse = mean_squared_error(y_val, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_val, y_pred, multioutput='raw_values')
    r2 = r2_score(y_val, y_pred, multioutput='raw_values')
    
    # Calculate additional metrics
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    if target_scaler is not None:
        rmse_orig = np.sqrt(mse_orig)
        
        return {
            'mse': mse.tolist(),
            'rmse': rmse.tolist(),
            'mae': mae.tolist(),
            'r2': r2.tolist(),
            'mse_orig': mse_orig.tolist(),
            'rmse_orig': rmse_orig.tolist(),
            'mae_orig': mae_orig.tolist(),
            'r2_orig': r2_orig.tolist()
        }
    else:
        return {
            'mse': mse.tolist(),
            'rmse': rmse.tolist(),
            'mae': mae.tolist(),
            'r2': r2.tolist()
        }

def plot_predictions(model, X_val, y_val, model_type, target_scaler=None, 
                     samples=100, save_dir='evaluation_plots'):
    """Plot predictions vs actual values
    
    Args:
        model (tf.keras.Model): The model to generate predictions
        X_val (np.ndarray): Validation input data
        y_val (np.ndarray): Validation target data
        model_type (str): Type of model
        target_scaler (object, optional): Scaler for inverting target normalization
        samples (int): Number of samples to plot
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate predictions
    y_pred = model.predict(X_val)
    
    # Use only a subset of samples for plotting
    indices = np.random.choice(len(y_val), min(samples, len(y_val)), replace=False)
    
    # Inverse transform if scaler is provided
    if target_scaler is not None:
        y_val = target_scaler.inverse_transform(y_val)
        y_pred = target_scaler.inverse_transform(y_pred)
    
    # Get output dimension
    output_dim = y_val.shape[1]
    
    # Plot each output dimension
    for i in range(output_dim):
        plt.figure(figsize=(10, 6))
        
        # Plot actual vs predicted
        plt.scatter(y_val[indices, i], y_pred[indices, i], alpha=0.5)
        
        # Plot perfect prediction line
        min_val = min(np.min(y_val[indices, i]), np.min(y_pred[indices, i]))
        max_val = max(np.max(y_val[indices, i]), np.max(y_pred[indices, i]))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f'{model_type} - Output {i} - Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        
        plt.savefig(os.path.join(save_dir, f'{model_type}_output{i}_pred_vs_actual.png'))
        plt.close()
        
    # Plot sequence of predictions for time series models
    if len(X_val.shape) == 3:  # Check if it's a sequence model
        # Take first 100 sequences for plotting
        plot_samples = min(100, len(y_val))
        
        for i in range(output_dim):
            plt.figure(figsize=(12, 6))
            
            plt.plot(y_val[:plot_samples, i], label='Actual')
            plt.plot(y_pred[:plot_samples, i], label='Predicted')
            
            plt.title(f'{model_type} - Output {i} - Time Series Prediction')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(os.path.join(save_dir, f'{model_type}_output{i}_timeseries.png'))
            plt.close()

def evaluate_all_models(data_file='telemetry.csv', sequence_length=10, use_best=True):
    """Evaluate all trained models
    
    Args:
        data_file (str): Path to the telemetry data file
        sequence_length (int): Sequence length for time series data
        use_best (bool): Whether to use the best checkpoint models
        
    Returns:
        dict: Evaluation results for all models
    """
    # Create directories
    os.makedirs('evaluation_results', exist_ok=True)
    os.makedirs('evaluation_plots', exist_ok=True)
    
    # Initialize data processor
    processor = TelemetryDataProcessor()
    
    # Prepare training/validation data
    print("Preparing validation data...")
    train_val_splits = processor.prepare_training_data(
        filename=data_file,
        sequence_length=sequence_length
    )
    
    # Model types
    model_types = [
        'high_level',
        'tactical',
        'low_level',
        'gear_selection',
        'corner_handling'
    ]
    
    # Store evaluation results
    evaluation_results = {}
    
    # Get target variable names for each model
    target_names = {
        'high_level': processor.target_groups['high_level'],
        'tactical': processor.target_groups['tactical'],
        'low_level': processor.target_groups['low_level'],
        'gear_selection': processor.target_groups['gear_selection'],
        'corner_handling': processor.target_groups['corner_handling']
    }
    
    # Evaluate each model
    for model_type in model_types:
        print(f"\nEvaluating {model_type} model...")
        
        # Get validation data for this model
        X_val = train_val_splits[model_type]['X_val']
        y_val = train_val_splits[model_type]['y_val']
        
        try:
            # Load model
            model = load_model(model_type, use_best=use_best)
            
            # Load scalers
            feature_scaler, target_scaler = load_scalers(model_type)
            
            # Evaluate model
            metrics = evaluate_model(model, X_val, y_val, target_scaler)
            
            # Create a combined metrics dictionary with target names
            named_metrics = {}
            for metric_name, values in metrics.items():
                if isinstance(values, list):
                    named_metrics[metric_name] = {
                        target_names[model_type][i]: values[i] 
                        for i in range(min(len(target_names[model_type]), len(values)))
                    }
                else:
                    named_metrics[metric_name] = values
            
            # Store results
            evaluation_results[model_type] = named_metrics
            
            # Plot predictions
            plot_predictions(model, X_val, y_val, model_type, target_scaler)
            
        except Exception as e:
            print(f"Error evaluating {model_type} model: {e}")
            evaluation_results[model_type] = {"error": str(e)}
    
    # Save overall evaluation results
    with open('evaluation_results/evaluation_summary.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    # Save a more readable text report
    with open('evaluation_results/evaluation_report.txt', 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("=====================\n\n")
        
        for model_type, metrics in evaluation_results.items():
            f.write(f"{model_type.upper()} MODEL\n")
            f.write("-" * len(f"{model_type.upper()} MODEL") + "\n")
            
            if "error" in metrics:
                f.write(f"Error: {metrics['error']}\n\n")
                continue
                
            # Write normalized metrics
            f.write("Normalized Metrics:\n")
            for target_name, r2_value in metrics.get('r2', {}).items():
                f.write(f"  {target_name}:\n")
                f.write(f"    R² Score: {r2_value:.4f}\n")
                f.write(f"    MAE: {metrics['mae'][target_name]:.4f}\n")
                f.write(f"    RMSE: {metrics['rmse'][target_name]:.4f}\n")
            
            # Write original scale metrics if available
            if 'r2_orig' in metrics:
                f.write("\nOriginal Scale Metrics:\n")
                for target_name, r2_value in metrics['r2_orig'].items():
                    f.write(f"  {target_name}:\n")
                    f.write(f"    R² Score: {r2_value:.4f}\n")
                    f.write(f"    MAE: {metrics['mae_orig'][target_name]:.4f}\n")
                    f.write(f"    RMSE: {metrics['rmse_orig'][target_name]:.4f}\n")
            
            f.write("\n\n")
    
    print("\nEvaluation completed!")
    print("Results saved in evaluation_results/evaluation_summary.json")
    print("Readable report saved in evaluation_results/evaluation_report.txt")
    print("Evaluation plots saved in evaluation_plots/ directory")
    
    return evaluation_results

def compare_model_performance():
    """Generate a comparison plot of model performance metrics"""
    try:
        # Load evaluation results
        with open('evaluation_results/evaluation_summary.json', 'r') as f:
            results = json.load(f)
        
        # Create a plot to compare R² scores across models
        plt.figure(figsize=(12, 8))
        
        model_types = list(results.keys())
        x = np.arange(len(model_types))
        width = 0.2
        
        # Group by output variable and plot grouped bars
        all_outputs = set()
        for model_type in model_types:
            if 'error' not in results[model_type] and 'r2_orig' in results[model_type]:
                all_outputs.update(results[model_type]['r2_orig'].keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_outputs)))
        
        for i, output in enumerate(all_outputs):
            r2_values = []
            for model_type in model_types:
                if ('error' not in results[model_type] and 
                    'r2_orig' in results[model_type] and 
                    output in results[model_type]['r2_orig']):
                    r2_values.append(results[model_type]['r2_orig'][output])
                else:
                    r2_values.append(0)  # Use 0 for missing values
            
            plt.bar(x + (i - len(all_outputs)/2 + 0.5) * width, r2_values, 
                   width, label=output, color=colors[i])
        
        plt.xlabel('Model Type')
        plt.ylabel('R² Score (Original Scale)')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_types, rotation=45)
        plt.legend(title='Output Variable')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('evaluation_results/model_comparison.png')
        plt.close()
        
        print("Model comparison plot saved to evaluation_results/model_comparison.png")
        
    except Exception as e:
        print(f"Error generating comparison plot: {e}")

if __name__ == '__main__':
    # Evaluate all models (default: use best checkpoints)
    results = evaluate_all_models(use_best=True)
    
    # Generate comparison plots
    compare_model_performance()