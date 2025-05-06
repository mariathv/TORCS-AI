from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_and_preprocess_data
import os
import matplotlib.pyplot as plt

def build_and_train_model(X_train, X_test, y_train, y_test, epochs=50, batch_size=32, model_save_path='controller/model'):
    """Build and train a model optimized for 13 features"""
    print(f"Building model for {X_train.shape[1]} input features (reduced feature set)")
    
    # ----- Build a smaller, faster neural network -------
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))   # Smaller first layer
    model.add(Dense(32, activation='relu'))                               # Smaller hidden layer
    model.add(Dense(y_train.shape[1], activation='linear'))               # Output layer
    
    # ---- compile the model with slightly higher learning rate for faster convergence
    model.compile(optimizer=Adam(learning_rate=0.002),
                  loss='mse', 
                  metrics=['mae'])
    
    # Print model summary
    model.summary()
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")

    # ---- train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # ---- evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
    
    # ---- save the model
    # Add .keras extension if no extension is present
    if not model_save_path.endswith(('.keras', '.h5')):
        model_save_path += '.keras'
    
    # Only try to create directory if there's a directory component in the path
    dir_path = os.path.dirname(model_save_path)
    if dir_path:  # Only create directory if dir_path is not empty
        os.makedirs(dir_path, exist_ok=True)
    save_model(model, model_save_path)
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
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
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
    # Add .keras extension if no extension is present
    if not model_path.endswith(('.keras', '.h5')):
        model_path += '.keras'
        
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        print(f"Model input shape: {model.input_shape}, Expected features: {model.input_shape[1]}")
        return model
    else:
        print(f"No model found at {model_path}")
        return None

