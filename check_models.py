import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import json

def custom_mse():
    """Custom MSE function that can be properly serialized"""
    def mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    return mse

def check_model_scaler_compatibility(models_dir='models', scalers_dir='model_scalers'):
    """Check compatibility between models and scalers"""
    model_types = [
        'high_level',
        'tactical',
        'low_level',
        'gear_selection',
        'corner_handling'
    ]
    
    print("Checking model and scaler compatibility...")
    print("=" * 60)
    
    for model_type in model_types:
        print(f"\nModel type: {model_type}")
        print("-" * 40)
        
        # Check if model exists
        model_path_keras = os.path.join(models_dir, f'{model_type}_model.keras')
        model_path_h5 = os.path.join(models_dir, f'{model_type}_model.h5')
        model_path_dir = os.path.join(models_dir, f'{model_type}_model')
        
        model = None
        if os.path.exists(model_path_keras):
            try:
                model = keras.models.load_model(
                    model_path_keras, 
                    custom_objects={'mse': custom_mse()}
                )
                print(f"Loaded {model_type} model from Keras file")
            except Exception as e:
                print(f"Error loading {model_type} model from Keras: {e}")
        elif os.path.exists(model_path_h5):
            try:
                model = keras.models.load_model(
                    model_path_h5, 
                    custom_objects={'mse': custom_mse()}
                )
                print(f"Loaded {model_type} model from H5 file")
            except Exception as e:
                print(f"Error loading {model_type} model from H5: {e}")
        elif os.path.exists(model_path_dir) and os.path.isdir(model_path_dir):
            try:
                model = keras.models.load_model(
                    model_path_dir,
                    custom_objects={'mse': custom_mse()}
                )
                print(f"Loaded {model_type} model from SavedModel directory")
            except Exception as e:
                print(f"Error loading {model_type} model from directory: {e}")
        else:
            print(f"Model not found at {model_path_keras}, {model_path_h5} or {model_path_dir}")
        
        # Check model input shape
        if model:
            input_shape = model.input_shape
            output_shape = model.output_shape
            print(f"Model input shape: {input_shape}")
            print(f"Model output shape: {output_shape}")
        
        # Check if scaler exists
        scaler_path = os.path.join(scalers_dir, f'{model_type}_scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"Loaded {model_type} scaler")
                
                # Check scaler shape
                if hasattr(scaler, 'n_features_in_'):
                    print(f"Scaler expects {scaler.n_features_in_} features")
                elif hasattr(scaler, 'mean_'):
                    print(f"Scaler mean shape: {scaler.mean_.shape}")
                else:
                    print("Scaler shape information not available")
            except Exception as e:
                print(f"Error loading {model_type} scaler: {e}")
        else:
            print(f"Scaler not found at {scaler_path}")
    
    print("\nRecommended feature count for prepare_state_data method:")
    print("=" * 60)
    print("Based on this analysis, ensure your prepare_state_data method in model_coordinator.py")
    print("returns a feature vector with the correct number of features for each model.")

if __name__ == "__main__":
    check_model_scaler_compatibility() 