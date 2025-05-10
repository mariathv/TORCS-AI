import os
import numpy as np
from data_processor import TelemetryDataProcessor
from models import create_model, custom_mse
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import pickle

def plot_training_history(history, model_name, save_dir='training_plots'):
    """Plot and save training and validation loss and MAE history."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 4))

    # Plot Loss
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
    """Save model summary and config as text and JSON files."""
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
    """Train all defined models using the telemetry data."""
    # Ensure output directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)
    os.makedirs('model_info', exist_ok=True)
    os.makedirs('model_scalers', exist_ok=True)

    # Initialize and prepare data
    processor = TelemetryDataProcessor()
    print("Preparing training data...")
    train_val_splits = processor.prepare_training_data(
        filename=data_file,
        sequence_length=sequence_length
    )

    # Define model types to train
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

        # Prepare model input/output shapes
        input_shape = (sequence_length, X_train.shape[-1])
        output_shape = y_train.shape[-1]

        # Create and compile the model
        model = create_model(model_type, input_shape, output_shape)
        model.compile_model()

        # Save architecture and configuration
        save_model_info(model.model, model_type, input_shape, output_shape)

        # Define callbacks
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'models/{model_type}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        # Train the model
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=100,
            batch_size=32,
            callbacks=[model_checkpoint, early_stopping]
        )

        # Plot and save training history
        plot_training_history(history, model_type)

        # Save the full model (including custom objects)
        model.save(f'models/{model_type}_model.keras')

        # Save the scaler if available
        if hasattr(processor, 'scalers') and model_type in processor.scalers:
            scaler_path = f'model_scalers/{model_type}_scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(processor.scalers[model_type], f)
            print(f"Saved scaler for {model_type} model.")

        # Store final metrics
        training_results[model_type] = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_train_mae': history.history['mae'][-1],
            'final_val_mae': history.history['val_mae'][-1]
        }

    # Save all training metrics
    with open('training_results.json', 'w', encoding='utf-8') as f:
        json.dump(training_results, f, indent=4)

    print("\n✅ Training completed!")
    print("🔒 Model checkpoints saved in models/")
    print("📈 Training plots saved in training_plots/")
    print("📄 Model info saved in model_info/")
    print("📊 Training results saved in training_results.json")

if __name__ == '__main__':
    train_all_models()
