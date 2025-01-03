from flask import Flask, render_template, request, redirect, flash, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import pandas as pd
import os
import sqlite3
from net_teach import train_and_predict_lstm  # Импортируем функцию обучения нейросети
from flask_socketio import SocketIO, emit
from threading import Thread
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


app = Flask(__name__)
socketio = SocketIO(app)
app.config['UPLOAD_FOLDER'] = 'C:/Users/Сергей/Desktop/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SECRET_KEY'] = 'your_secret_key'
db = SQLAlchemy(app)

# Модель данных для таблицы
class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    total_kdsi = db.Column(db.Float, nullable=False)
    aaf = db.Column(db.Float, nullable=False)
    rely = db.Column(db.Float, nullable=False)
    data = db.Column(db.Float, nullable=False)
    cplx = db.Column(db.Float, nullable=False)
    time = db.Column(db.Float, nullable=False)
    stor = db.Column(db.Float, nullable=False)
    virt = db.Column(db.Float, nullable=False)
    turn = db.Column(db.Float, nullable=False)
    acap = db.Column(db.Float, nullable=False)
    aexp = db.Column(db.Float, nullable=False)
    pcap = db.Column(db.Float, nullable=False)
    vexp = db.Column(db.Float, nullable=False)
    lexp = db.Column(db.Float, nullable=False)
    modp = db.Column(db.Float, nullable=False)
    tool = db.Column(db.Float, nullable=False)
    sced = db.Column(db.Float, nullable=False)
    effort = db.Column(db.Float, nullable=False)

def __init__(self, total_kdsi, aaf, rely, data, cplx, time, stor, virt, turn, acap, aexp, pcap, vexp, lexp, modp, tool, sced, effort):
    self.total_kdsi = total_kdsi
    self.aaf = aaf
    self.rely = rely
    self.data = data
    self.cplx = cplx
    self.time = time
    self.stor = stor
    self.virt = virt
    self.turn = turn
    self.acap = acap
    self.aexp = aexp
    self.pcap = pcap
    self.vexp = vexp
    self.lexp = lexp
    self.modp = modp
    self.tool = tool
    self.sced = sced
    self.effort = effort

# Главная страница с формой загрузки файла
@app.route('/')
def start():
    return render_template('start.html')

# Обработчик загрузки файла
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if 'file' not in request.files:
        flash('Нет файла для загрузки')
        return render_template('upload.html')

    file = request.files['file']

    if file.filename == '':
        flash('Файл не выбран')
        return redirect(request.url)

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Чтение CSV файла
        data = pd.read_csv(file_path, delimiter=',')

        # Загрузка данных в базу данных
        for index, row in data.iterrows():
            new_record = Data(total_kdsi=row['total_kdsi'],
                              aaf=row['aaf'],
                              rely=row['rely'],
                              data=row['data'],
                              cplx=row['cplx'],
                              time=row['time'],
                              stor=row['stor'],
                              virt=row['virt'],
                              turn=row['turn'],
                              acap=row['acap'],
                              aexp=row['aexp'],
                              pcap=row['pcap'],
                              vexp=row['vexp'],
                              lexp=row['lexp'],
                              modp=row['modp'],
                              tool=row['tool'],
                              sced=row['sced'],
                              effort=row['effort'])
            db.session.add(new_record)

        db.session.commit()
        flash('Файл успешно загружен и данные сохранены в базе')
        return redirect(url_for('start'))

    flash('Неверный формат файла. Только CSV.')

    return render_template('upload.html')

@app.route("/project_data")
def project_data():
    return render_template('project_data.html')

@app.route("/uploaded_data")
def uploaded_data():
    # Подключение к базе данных SQLite
    conn = sqlite3.connect('instance/data.db')
    df_data = pd.read_sql_query("SELECT * FROM data", conn)
    conn.close()
    return render_template('uploaded_data.html', varible=df_data)

@app.route("/count", methods=['GET', 'POST'])
def count():
    if request.method == 'GET':
        return render_template('count.html')

    if request.method == 'POST':
        model = load_model('model.keras')
        # Получаем данные от пользователя
        input_data = request.get_json()
        print(input_data)
        # Преобразуем входные данные в формат numpy для модели
        data = np.array([[
            input_data['total_kdsi'],
            input_data['aaf'],
            input_data['rely'],
            input_data['data'],
            input_data['cplx'],
            input_data['time'],
            input_data['stor'],
            input_data['virt'],
            input_data['turn'],
            input_data['acap'],
            input_data['aexp'],
            input_data['pcap'],
            input_data['vexp'],
            input_data['lexp'],
            input_data['modp'],
            input_data['tool'],
            input_data['sced']
        ]])

        # Масштабирование данных (как при обучении)
        scaler_X = StandardScaler()
        data_scaled = scaler_X.fit_transform(data)

        # Преобразование данных в формат (samples, timesteps, features) для LSTM
        data_scaled = np.reshape(data_scaled, (data_scaled.shape[0], 1, data_scaled.shape[1]))

        # Предсказание с использованием модели
        result = model.predict(data_scaled)

        # Возвращаем результат на страницу
        return jsonify({'result': float(result[0][0])})


# Маршрут для обучения модели
@app.route("/teaching")
def teaching():
    return render_template('teaching.html')

@app.route("/start_training", methods=["POST"])
def start_training():
    # Подключаемся к базе данных и получаем данные для обучения
    conn = sqlite3.connect('instance/data.db')
    df_data = pd.read_sql_query("SELECT * FROM data", conn)
    conn.close()

    # Запускаем обучение в отдельном потоке, чтобы не блокировать основной поток
    thread = Thread(target=train_and_predict_lstm, args=(df_data, socketio))
    thread.start()

    # Немедленный ответ клиенту
    return jsonify({'message': 'Процесс обучения начался'})

@app.route("/Signin", methods=['GET', 'POST'])
def Signin():
    # Ваш код логики входа
    return render_template('Signin.html')


if __name__ == '__main__':
    # Установка контекста приложения для работы с базой данных
    with app.app_context():
        db.create_all()  # Создание таблиц в базе данных
    # Запуск сервера Flask
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)


