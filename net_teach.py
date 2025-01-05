import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from flask_socketio import emit
import time
import joblib
from tensorflow.keras.layers import Input


def train_and_predict_lstm(data, socketio):
    """
    Функция для обучения модели LSTM и предсказания на основе входных данных.

    Параметры:
        data (pd.DataFrame): DataFrame, содержащий входные данные и целевую переменную.
        socketio: объект SocketIO для отправки данных на клиент.

    Возвращает:
        y_pred_original (np.ndarray): Предсказанные значения 'effort' в исходном масштабе.
    """

    # Выделение входных данных (X) и целевой переменной (y)
    X = data[['total_kdsi', 'aaf', 'rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn',
              'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced']].values
    y = data['effort'].values

    # Масштабирование данных
    # scaler_X = StandardScaler()
    X_scaled = X #scaler_X.fit_transform(X)

    # scaler_y = StandardScaler()
    y_scaled = y #scaler_y.fit_transform(y.reshape(-1, 1))

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Преобразуем данные в формат для LSTM (samples, timesteps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Создание модели LSTM с использованием Input() для явного задания формы входных данных
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(100, return_sequences=True),
        LSTM(100),
        Dense(1)
    ])

    # Компиляция модели
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Обучение модели с отправкой обновлений на клиент через SocketIO
    epochs = 85

    for epoch in range(epochs):
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=0)
        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]

        # Отправляем данные об ошибках на клиент через SocketIO
        # Логирование прогресса
        print(f"Эпоха {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        socketio.emit('training_update', {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})
        time.sleep(1)

    # Предсказание на тестовых данных
    y_pred = model.predict(X_test)
    # Сохранение scaler
    #joblib.dump(scaler_y, 'scaler_y.save')
    # Обратное преобразование масштабированных предсказаний в исходный масштаб
    #y_pred_original = scaler_y.inverse_transform(y_pred)

    # Сохранение обученной модели
    model.save('model.keras')

    return y_pred
