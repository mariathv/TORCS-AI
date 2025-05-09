import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import json
from data_processor import TelemetryDataProcessor
from models import create_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
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
    
    # Save model summary
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    
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
    
    # Enable mixed precision training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Configure TensorFlow for better performance
    tf.config.optimizer.set_jit(True)  # Enable XLA compilation
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': True,
        'constant_folding': True,
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
        'dependency_optimization': True,
        'loop_optimization': True,
        'function_optimization': True,
        'debug_stripper': True,
        'disable_model_pruning': False,
        'scoped_allocator_optimization': True,
        'pin_to_host_optimization': True,
        'implementation_selector': True,
        'auto_mixed_precision': True
    })
    
    # Initialize data processor
    processor = TelemetryDataProcessor()
    
    # Prepare training data
    logger.info("Preparing training data...")
    train_val_splits = processor.prepare_training_data(
        filename=data_file,
        sequence_length=sequence_length
    )
    
    # Train each model
    model_types = [
        'high_level',
        'tactical',
        'low_level',
        'corner_handling'
    ]
    
    training_results = {}
    
    for model_type in model_types:
        logger.info(f"\nTraining {model_type} model...")
        
        # Get data for this model
        data = train_val_splits[model_type]
        X_train, X_val = data['X_train'], data['X_val']
        y_train, y_val = data['y_train'], data['y_val']
        
        # Create and compile model
        input_shape = (sequence_length, X_train.shape[-1])
        output_shape = y_train.shape[-1]
        
        model = create_model(model_type, input_shape, output_shape)
        
        # Use a lower learning rate for more stable training
        model.compile_model(learning_rate=0.0005)
        
        # Save model information
        save_model_info(
            model.model,
            model_type,
            input_shape,
            output_shape
        )
        
        # Train model with optimized parameters
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=100,  # More epochs for better convergence
            batch_size=32  # Smaller batch size for better generalization
        )
        
        # Plot and save training history
        plot_training_history(history, model_type)
        
        # Save model
        model.save(f'models/{model_type}_model.keras')
        
        # Store results
        training_results[model_type] = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_train_mae': history.history['mean_absolute_error'][-1],
            'final_val_mae': history.history['val_mean_absolute_error'][-1]
        }
    
    # Save overall training results
    with open('training_results.json', 'w') as f:
        json.dump(training_results, f, indent=4)
    
    logger.info("\nTraining completed!")
    logger.info("Results saved in training_results.json")
    logger.info("Model checkpoints saved in models/ directory")
    logger.info("Training plots saved in training_plots/ directory")
    logger.info("Model information saved in model_info/ directory")

if __name__ == '__main__':
    train_all_models()