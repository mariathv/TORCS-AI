import os
import numpy as np
from data_processor import TelemetryDataProcessor
from models import create_model, custom_mse
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import pickle

def plot_training_history(history, model_name, save_dir='training_plots'):
    """Plot and save training history"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'))
    plt.close()

def save_model_info(model, model_name, input_shape, output_shape, save_dir='model_info'):
    """Save model information and architecture"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model summary to a string first
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    
    # Write the summary to file with UTF-8 encoding
    with open(os.path.join(save_dir, f'{model_name}_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_list))
    
    # Save model configuration
    config = {
        'model_name': model_name,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'total_params': model.count_params()
    }
    
    with open(os.path.join(save_dir, f'{model_name}_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def train_all_models(data_file='telemetry.csv', sequence_length=10):
    """Train all models using the telemetry data"""
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)
    os.makedirs('model_info', exist_ok=True)
    os.makedirs('model_scalers', exist_ok=True)
    
    # Initialize data processor
    processor = TelemetryDataProcessor()
    
    # Prepare training data
    print("Preparing training data...")
    train_val_splits = processor.prepare_training_data(
        filename=data_file,
        sequence_length=sequence_length
    )
    
    # Train each model
    model_types = [
        'high_level',
        'tactical',
        'low_level',
        'gear_selection',
        'corner_handling'
    ]
    
    training_results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        
        # Get data for this model
        data = train_val_splits[model_type]
        X_train, X_val = data['X_train'], data['X_val']
        y_train, y_val = data['y_train'], data['y_val']
        
        # Create and compile model
        input_shape = (sequence_length, X_train.shape[-1])
        output_shape = y_train.shape[-1]
        
        model = create_model(model_type, input_shape, output_shape)
        model.compile_model()
        
        # Save model information
        save_model_info(
            model.model,
            model_type,
            input_shape,
            output_shape
        )
        
        # Train model
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=100,
            batch_size=32
        )
        
        # Plot and save training history
        plot_training_history(history, model_type)
        
        # Save model with custom objects
        model.save(f'models/{model_type}_model.h5')
        
        # Save scaler if available
        if hasattr(processor, 'scalers') and model_type in processor.scalers:
            scaler_path = f'model_scalers/{model_type}_scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(processor.scalers[model_type], f)
            print(f"Saved scaler for {model_type} model")
        
        # Store results
        training_results[model_type] = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_train_mae': history.history['mae'][-1],
            'final_val_mae': history.history['val_mae'][-1]
        }
    
    # Save overall training results
    with open('training_results.json', 'w') as f:
        json.dump(training_results, f, indent=4)
    
    print("\nTraining completed!")
    print("Results saved in training_results.json")
    print("Model checkpoints saved in models/ directory")
    print("Training plots saved in training_plots/ directory")
    print("Model information saved in model_info/ directory")

if __name__ == '__main__':
    train_all_models() 