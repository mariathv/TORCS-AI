from data_preprocessing import load_and_preprocess_data
from model_training import build_and_train_model, load_trained_model
import argparse
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and manage TORCS ML controller')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data', type=str, default='../telemetry_logs/telemetry.csv', help='Path to telemetry data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model_path', type=str, default='model', help='Path to save/load model')
    parser.add_argument('--save_scaler', action='store_true', help='Save the data scaler for inference')
    
    args = parser.parse_args()
    
    # Ensure model path is valid and has consistent slashes
    model_path = args.model_path.replace('\\', '/')
    
    if args.train:
        print("=" * 50)
        print("TRAINING ML MODEL WITH REDUCED FEATURE SET (12 FEATURES)")
        print("This improves performance while maintaining control quality")
        print("=" * 50)
        
        print(f"Loading data from {args.data}")
        # Load and preprocess data
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(args.data, return_scaler=True)
        
        # Build and train the model
        model = build_and_train_model(
            X_train, X_test, y_train, y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=model_path
        )
        
        # Save the scaler for inference
        if args.save_scaler:
            # Add .keras extension if needed for determining the model path 
            if not model_path.endswith(('.keras', '.h5')):
                actual_model_path = model_path + '.keras'
            else:
                actual_model_path = model_path
                
            # Create directory for scaler if needed
            dir_path = os.path.dirname(actual_model_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # Get base path without extension for scaler
            base_path = actual_model_path.rsplit('.', 1)[0]
            scaler_path = f"{base_path}_scaler.pkl"
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved to {scaler_path}")
            
            # Save feature count information in a text file for reference
            info_path = f"{base_path}_info.txt"
            with open(info_path, 'w') as f:
                f.write(f"Feature count: {X_train.shape[1]}\n")
                f.write("Using reduced feature set (12 features) for better performance\n")
                f.write("- Speed components (3): speedX, speedY, speedZ\n")
                f.write("- Position data (2): angle, trackPos\n")
                f.write("- Engine state (2): rpm, gear\n")
                f.write("- Track sensors (5): front, front-left, front-right\n")
            print(f"Model info saved to {info_path}")
    else:
        # Just load the model to verify it exists
        model = load_trained_model(model_path)
        if model:
            print("=" * 50)
            print("Model loaded successfully. Ready for inference.")
            print(f"Model input shape: {model.input_shape}")
            
            if model.input_shape[1] == 12:
                print("USING OPTIMIZED MODEL with 12 features (should be faster)")
            else:
                print(f"NOTE: Model uses {model.input_shape[1]} features (not the optimized 12)")
            print("=" * 50)

if __name__ == "__main__":
    main()
