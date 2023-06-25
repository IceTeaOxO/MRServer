from flask import Flask, render_template, request, jsonify

import json
from keras.models import load_model
import numpy as np
import time
from threading import Thread

# 前置設定
app = Flask(__name__)
app = Flask(__name__, static_folder='static')

# 暫存資料
data_list = []
number_list = []
speech_list = []

def add_number():
    num = 0
    while True:
        time.sleep(5)
        number_list.append(num)
        num += 1
@app.route('/test', methods=['GET'])
def get_test():
    return jsonify(number_list)


# 根目錄，測試用
@app.route('/')
def index():
    return render_template('index.html')
# 取得語音辨識結果
@app.route('/speech', methods=['POST'])
def SpeechResult():
    speech = request.form.get('speech')
    # 在這裡進行資料處理或其他操作
    print("Received speech:", speech)
    speech_list.append(speech)
    return speech_list

@app.route('/speech', methods=['GET'])
def get_SpeechResult():
    return jsonify(speech_list)

# 取得手語辨識結果
@app.route('/RR', methods=['POST'])
def RecongResult():
    value = request.form.get('value')
    data_list.append(value)
    return data_list

@app.route('/RR', methods=['GET'])
def get_RecongResult():
    return jsonify(data_list)

# 取得語序排序結果
@app.route('/WO')
def WordOrder():
    return render_template('index.html')


if __name__ == '__main__':
    thread = Thread(target=add_number)
    thread.start()
    app.run(host="127.0.0.1", port=8080, debug=True)