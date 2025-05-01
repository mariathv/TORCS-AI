from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_and_preprocess_data

def build_and_train_model(X_train, X_test, y_train, y_test):
    # ----- Build the neural network -------
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # Input layer
    model.add(Dense(64, activation='relu'))  # ---- Hidden layer
    model.add(Dense(32, activation='relu'))  # ---- Hidden layer
    model.add(Dense(y_train.shape[1], activation='linear'))  # ----- Output layer

    # ---- compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse', 
                  metrics=['mae'])

    # ---- train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # ---- evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    return model

