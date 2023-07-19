from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import json
from keras.models import load_model
import numpy as np
import time
from threading import Thread
import urllib.parse
# 前置設定
app = Flask(__name__)
app = Flask(__name__, static_folder='static')
scheduler = BackgroundScheduler()
# 全域變數負責儲存資料
data_list = [] # 儲存手語辨識結果
number_list = [] # 儲存假資料
speech_list = [] # 儲存語音辨識結果
trans_list = [] # 儲存語序辨識結果
alarm_set = False
last_updated_time = 0

# 載入語序模型
data_path_trans = 'EngToChinese.txt'
input_texts = []
target_texts = []

input_characters = set()
target_characters = set()
with open(data_path_trans, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines:
    input_text, target_text= line.split('   ')
    # 用tab作用序列的开始，用\n作为序列的结束
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
        
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

# 对字符进行排序
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

# 计算共用到了什么字符
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
# 计算出最长的序列是多长
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# 建立字母到数字的映射
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
# 求數字到字母的映射
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())
        

model_trans = load_model("model071604-20.h5")

# 辨識語序方法
def translate(model_opt='check yes paper sign'):
    
    in_encoder = np.zeros((1, max_encoder_seq_length, num_encoder_tokens),dtype='float32')

    for t, char in enumerate(model_opt):
        in_encoder[0, t, input_token_index[char]] = 1.
    in_encoder[0, t + 1:, input_token_index[' ']] = 1.

    in_decoder = np.zeros((len(in_encoder), max_decoder_seq_length, num_decoder_tokens),dtype='float32')
    in_decoder[:, 0, target_token_index["\t"]] = 1

    # 生成 decoder 的 output
    for i in range(max_decoder_seq_length - 1):
        predict = model_trans.predict([in_encoder, in_decoder])
        predict = predict.argmax(axis=-1)
        predict_ = predict[:, i].ravel().tolist()
        for j, x in enumerate(predict_):
            in_decoder[j, i + 1, x] = 1 # 將每個預測出的 token 設為 decoder 下一個 timestsmp 的輸入

    seq_index = 0
    decoded_sentence = ""
    output_seq = predict[seq_index, :].ravel().tolist()
    for x in output_seq:
        if reverse_target_char_index[x] == "\n":
            break
        else:
            decoded_sentence+=reverse_target_char_index[x]

    # print('Input sentence:', model_opt)
    # print('Decoded sentence:', decoded_sentence)
    return decoded_sentence
def check_timeout():
    global alarm_set  # 声明全局变量
    global last_updated_time  # 声明全局变量
    global data_list
    global trans_list
    # 每過一秒就判斷一次
    # 滿足條件就翻譯語序
    if (alarm_set and (abs(time.time() - last_updated_time) >= 10)):
        # print(time.time())    
        # 修改DATA型態
        data_result = ' '.join(data_list)#將[1,2]的資料型態轉為"1 2"
        # 判斷語序
        trans_result = translate(data_result)#' '.join(sentence)
        print('---result---', trans_result)
        # 將結果存在trans_list中
        trans_result = urllib.parse.unquote(trans_result)
        trans_list.append(trans_result)
        
        # 清空 data_list 資料
        data_list = []
        #重新計時
        last_updated_time = time.time()
        # 將通知取消
        alarm_set = False





# unity測試用
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

# 語音辨識模組將結果丟上server
@app.route('/speech', methods=['POST'])
def SpeechResult():
    '''
    POST範例
    data = {
                'speech': speech_recognition_result.text
            }
    '''
    speech = request.form.get('speech')
    # 這裡進行資料處理或其他操作
    print("Received speech:", speech)
    speech = urllib.parse.unquote(speech)
    # 將POST的資料儲存進全域變數
    speech_list.append(speech)
    return speech_list

# 取得語音辨識結果
@app.route('/speech', methods=['GET'])
def get_SpeechResult():
    # 將全域變數以json的格式回傳
    return jsonify(speech_list)


# 手語辨識模組將結果丟上server
@app.route('/RR', methods=['POST'])
def RecongResult():
    '''
    POST範例
    data = {
        'value': data
        }
    '''
    value = request.form.get('value')
    value = urllib.parse.unquote(value)
    # 將POST資料存進全域變數
    data_list.append(value)
    # 如果有新的手語儲存，就設alarm_set = True，並更新last_updated_time
    global alarm_set
    alarm_set = True
    print("set alarm_set TRue")
    global last_updated_time
    last_updated_time = time.time()
    print(last_updated_time,"updated time")
    return data_list

# 取得手語辨識結果
@app.route('/RR', methods=['GET'])
def get_RecongResult():
    '''
    以json的格式回傳手語辨識資料
    資料格式
    [
    "sign",
    "0"
    ]
    '''
    return jsonify(data_list)

# 儲存語序排序結果
@app.route('/TL', methods=['POST'])
def Translate():
    
    
    return trans_list

# 取得語序排序結果
@app.route('/TL', methods=['GET'])
def get_Translate():
    
    return jsonify(trans_list)



if __name__ == '__main__':
    thread = Thread(target=add_number)
    thread.start()
    scheduler.add_job(check_timeout, 'interval', seconds=1)  # 每1秒触发一次定时器
    scheduler.start()
    app.run(host="127.0.0.1", port=8080, debug=True)