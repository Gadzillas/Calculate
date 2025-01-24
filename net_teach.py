import numpy as np
from keras import Input, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from flask_socketio import emit
import tensorflow as tf
import time
import os
import pandas as pd

def train_and_predict_lstm(data, socketio, epochs=72, batch_size=32, scaler_y=None):
    """
    Train LSTM model and make predictions.

    Args:
        data (pd.DataFrame): Input data containing features and target.
        socketio: SocketIO object for client communication.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        np.ndarray: Predicted values for test data.
    """
    try:
        # Data preparation
        X = data[['total_kdsi', 'aaf', 'rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn',
                'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced']].values
        y = data['effort'].values

        # Scale data
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        # Create model with optimized architecture
        inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
        lstm1 = LSTM(128, return_sequences=True)(inputs)
        dropout1 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(64, return_sequences=True)(dropout1)
        dropout2 = Dropout(0.2)(lstm2)
        lstm3 = LSTM(32)(dropout2)  # Last LSTM layer has return_sequences=False by default
        dropout3 = Dropout(0.2)(lstm3)
        dense1 = Dense(16, activation='relu')(dropout3)
        outputs = Dense(1, activation='linear')(dense1)
        
        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Training loop with progress updates
        for epoch in range(epochs):
            history = model.fit(
                X_train, y_train,
                epochs=1,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            train_loss = history.history['loss'][0]
            val_loss = history.history['val_loss'][0]

            print(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            socketio.emit('training_update', {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            if early_stopping.stopped_epoch:
                print("Early stopping triggered")
                break

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_original = scaler_y.inverse_transform(y_pred)

        # Save model
        model.save('model.keras')
        
        return y_pred_original

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise