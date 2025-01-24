import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Загрузка обученной модели
model = load_model('model.keras')

# Пример входных данных (замените на реальные данные)
X_sample = np.array([[3.061, 0.84, 1.15, 1.08, 1, 1.11, 1.21, 0.87, 0.94, 0.71, 0.91, 1, 1, 1, 0.91, 0.91, 1
                      ]]).reshape(1, 1, -1)  # Размер (samples, timesteps, features)

# Сохранение весов каждого слоя LSTM
for i, layer in enumerate(model.layers):
    if 'lstm' in layer.name:  # Проверяем, что это LSTM слой
        weights = layer.get_weights()
        W_x = weights[0]  # Входные веса
        W_h = weights[1]  # Рекуррентные веса
        b = weights[2]  # Смещения

        # Сохранение весов в CSV
        np.savetxt(f"W_x_layer_{i}.csv", W_x, delimiter=",")
        np.savetxt(f"W_h_layer_{i}.csv", W_h, delimiter=",")
        np.savetxt(f"b_layer_{i}.csv", b, delimiter=",")


def get_lstm_internal_states(model, X_sample):
    """
    Извлечение внутренних состояний LSTM слоя (i_t, f_t, o_t, C_t, h_t).

    Параметры:
        model: обученная модель с LSTM слоями.
        X_sample: входные данные (один пример).

    Возвращает:
        states: словарь с промежуточными результатами (i_t, f_t, o_t, C_t, h_t).
    """
    # Найти первый LSTM слой
    lstm_layer = None
    for layer in model.layers:
        if 'lstm' in layer.name:
            lstm_layer = layer
            break

    if lstm_layer is None:
        raise ValueError("Модель не содержит LSTM слоев.")

    # Создать модель, возвращающую выходы LSTM слоя
    lstm_model = K.function([model.input], [lstm_layer.output, lstm_layer.states[0], lstm_layer.states[1]])

    # Прогнать входные данные через LSTM слой
    lstm_outputs, h_t, C_t = lstm_model([X_sample])

    # Веса и смещения LSTM слоя
    W_x, W_h, b = lstm_layer.get_weights()
    units = W_x.shape[1] // 4  # Число LSTM блоков

    # Разделение весов и смещений на группы
    W_xi, W_xf, W_xo, W_xc = np.split(W_x, 4, axis=1)
    W_hi, W_hf, W_ho, W_hc = np.split(W_h, 4, axis=1)
    b_i, b_f, b_o, b_c = np.split(b, 4)

    # Вычисления вентилей (на основе TensorFlow формул)
    x_t = X_sample[0, 0, :]  # Вход для текущего временного шага
    i_t = sigmoid(np.dot(x_t, W_xi) + np.dot(h_t, W_hi) + b_i)
    f_t = sigmoid(np.dot(x_t, W_xf) + np.dot(h_t, W_hf) + b_f)
    o_t = sigmoid(np.dot(x_t, W_xo) + np.dot(h_t, W_ho) + b_o)
    C_tilda_t = np.tanh(np.dot(x_t, W_xc) + np.dot(h_t, W_hc) + b_c)

    # Итоговые состояния памяти
    C_t = f_t * C_t + i_t * C_tilda_t
    h_t = o_t * np.tanh(C_t)

    return {
        "i_t": i_t,
        "f_t": f_t,
        "o_t": o_t,
        "C_tilda_t": C_tilda_t,
        "C_t": C_t,
        "h_t": h_t
    }


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Получение внутренних состояний для одного примера
states = get_lstm_internal_states(model, X_sample)

# Сохранение результатов в CSV
for state_name, state_value in states.items():
    np.savetxt(f"{state_name}.csv", state_value, delimiter=",")
