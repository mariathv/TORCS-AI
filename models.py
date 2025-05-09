import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all models with common functionality"""
    
    def __init__(self, input_shape, output_shape, model_name):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_name = model_name
        self.model = self._build_model()
    
    def _build_model(self):
        raise NotImplementedError("Subclasses must implement _build_model")
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    f'models/{self.model_name}_best.keras',
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
        )
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filepath):
        if not filepath.endswith('.keras'):
            filepath = filepath.replace('.h5', '.keras')
            if not filepath.endswith('.keras'):
                filepath += '.keras'
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        if filepath.endswith('.h5'):
            filepath = filepath.replace('.h5', '.keras')
        return tf.keras.models.load_model(filepath)

class HighLevelModel(BaseModel):
    """High-level planning model for strategic decisions"""
    
    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # LSTM layers for sequence processing
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.LSTM(64, return_sequences=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Dense layers for control output
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer with tanh activation for bounded outputs
        outputs = layers.Dense(self.output_shape, activation='tanh')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='high_level_model')

class TacticalModel(BaseModel):
    """Tactical control model for medium-term decisions"""
    
    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # Convolutional layers for feature extraction
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # LSTM layer for temporal dependencies
        x = layers.LSTM(64, return_sequences=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Dense layers for control output
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer with tanh activation
        outputs = layers.Dense(self.output_shape, activation='tanh')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='tactical_model')

class LowLevelModel(BaseModel):
    """Low-level control model for immediate actions"""
    
    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # Convolutional layers for immediate feature extraction
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # LSTM layer for short-term temporal dependencies
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Dense layers for precise control
        x = layers.Dense(16, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer with tanh activation
        outputs = layers.Dense(self.output_shape, activation='tanh')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='low_level_model')

class GearSelectionModel(BaseModel):
    """Model for gear selection decisions"""
    
    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # Simplified dense layers
        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.Dropout(0.1)(x)
        
        # Output layer with softmax for gear selection
        outputs = layers.Dense(self.output_shape, activation='softmax')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='gear_selection_model')

class CornerHandlingModel(BaseModel):
    """Model for corner entry/exit handling"""
    
    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # LSTM layers for corner sequence processing
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Dense layers for corner-specific control
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer with tanh activation
        outputs = layers.Dense(self.output_shape, activation='tanh')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='corner_handling_model')

def create_model(model_type, input_shape, output_shape):
    """Factory function to create the appropriate model"""
    model_classes = {
        'high_level': HighLevelModel,
        'tactical': TacticalModel,
        'low_level': LowLevelModel,
        'gear_selection': GearSelectionModel,
        'corner_handling': CornerHandlingModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_classes[model_type](input_shape, output_shape, model_type)