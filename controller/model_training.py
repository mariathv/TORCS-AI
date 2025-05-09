from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError  # Import metrics explicitly
from tensorflow.keras.losses import MeanSquaredError as MSELoss
from data_preprocessing import load_and_preprocess_data
import os
import matplotlib.pyplot as plt

def build_and_train_model(X_train, X_test, y_train, y_test, epochs=50, batch_size=32, model_save_path='controller/model'):
    """Build and train a model optimized for 24 features"""
    print(f"Building model for {X_train.shape[1]} input features (full feature set)")
    
    # ----- Build neural network for full feature set -------
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))   # Larger first layer for more features
    model.add(Dense(64, activation='relu'))                                # Larger hidden layer
    model.add(Dense(32, activation='relu'))                                # Additional hidden layer
    model.add(Dense(y_train.shape[1], activation='linear'))                # Output layer
    
    # ---- compile the model with object instances instead of string aliases
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MSELoss(),  # Use actual MeanSquaredError object
        metrics=[MeanAbsoluteError()]  # Use actual MeanAbsoluteError object instead of 'mae'
    )
    
    # Print model summary
    model.summary()
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")

    # ---- train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # ---- evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
    
    # ---- save the model
    # Always use .keras extension as recommended (never .h5)
    if not model_save_path.endswith('.keras'):
        model_save_path = model_save_path.rsplit('.', 1)[0] if '.' in model_save_path else model_save_path
        model_save_path += '.keras'
    
    # Only try to create directory if there's a directory component in the path
    dir_path = os.path.dirname(model_save_path)
    if dir_path:  # Only create directory if dir_path is not empty
        os.makedirs(dir_path, exist_ok=True)
    
    # Save with explicit format
    save_model(model, model_save_path, save_format='keras')
    print(f"Model saved to {model_save_path}")
    
    # ---- plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Use mean_absolute_error instead of mae in history access
    mae_key = 'mean_absolute_error'
    val_mae_key = 'val_mean_absolute_error'
    
    if mae_key not in history.history:
        # Fallback to mae if mean_absolute_error is not found
        mae_key = 'mae'
        val_mae_key = 'val_mae'
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history[mae_key])
    plt.plot(history.history[val_mae_key])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    # Create same directory check for the plot
    plot_base_path = model_save_path.rsplit('.', 1)[0]  # Remove extension
    plot_path = f"{plot_base_path}_training_history.png"
    plt.savefig(plot_path)
    plt.close()

    return model

def load_trained_model(model_path='controller/model'):
    """Load a previously trained model for inference"""
    # Always use .keras extension
    if not model_path.endswith('.keras'):
        model_path = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
        model_path += '.keras'
        
    if os.path.exists(model_path):
        try:
            # Define custom objects for loading
            custom_objects = {
                'MeanSquaredError': MSELoss,
                'MeanAbsoluteError': MeanAbsoluteError
            }
            
            # Always load with custom objects to be safe
            model = load_model(model_path, custom_objects=custom_objects)
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
        
        print(f"Model input shape: {model.input_shape}, Expected features: {model.input_shape[1]}")
        return model
    else:
        print(f"No model found at {model_path}")
        return None