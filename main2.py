from flask import Flask, render_template, request, jsonify, redirect
from apscheduler.schedulers.background import BackgroundScheduler
import json
import numpy as np
import time
from threading import Thread
import urllib.parse
from keras.models import load_model
import subprocess
class Server:
    def __init__(self):
        self.app = Flask(__name__, static_folder='static')
        self.scheduler = BackgroundScheduler()
        
        self.data_list:list = []  # 儲存手語辨識即時結果
        self.trans_list:list = []  # 儲存語序辨識結果
        self.speech_list:list = []  # 儲存語音辨識結果
        self.histor_data_list:list = []# 儲存手語辨識的歷史資料

        self.alarm_set = False
        self.last_updated_time = 0

        # 初始化模型列表和當前選擇的模型
        # 語序辨識模型
        self.available_models = {"model1": "model/model071604-20.h5", "model2": "model/model0529-20.h5", "model3": "model/model2_0904.h5"}  # 你可以根據需要添加更多模型
        self.current_model_name = "model1"
        self.model_trans, self.input_token_index, self.target_token_index, self.reverse_target_char_index, self.max_encoder_seq_length, self.max_decoder_seq_length, self.num_encoder_tokens, self.num_decoder_tokens = self.load_trans_model(self.current_model_name)
        
        # 初始化腳本列表和當前選擇的腳本
        # 若有新的語序模型可添加在此列表中，開啟伺服器就預先載入
        # 手語辨識模型腳本
        self.available_hands_models = {"model1": "VideoRecognition.py", "model2": "VideoRecognition0924_s1.py", "model3": "VideoRecognition0924_s2.py"}  # 你可以根據需要添加更多模型
        self.current_hands_model_name = "model1"
        self.process = None
        
        self.speech_process = None
        # self.model_trans, self.input_token_index, self.target_token_index, self.reverse_target_char_index, self.max_encoder_seq_length, self.max_decoder_seq_length, self.num_encoder_tokens, self.num_decoder_tokens = self.load_trans_model()

        self.initialize_routes()
        self.run()
    def append_to_list(self, lst:list, item, max_length=25):
        lst.append(item)
        if len(lst) > max_length:
            lst.pop(0)
    def load_trans_model(self, model_name):
        # (Same as original code for loading the translation model)
        # ...
        # ============================
        # 載入選定的模型
        model_file = self.available_models[model_name]
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
                

        model_trans = load_model(model_file)########
        print("切換模型:", model_file)
        # ===========================================
        return model_trans, input_token_index, target_token_index, reverse_target_char_index, max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens

    def switch_model(self, model_name):
        # 更新當前選擇的模型並重新載入
        self.current_model_name = model_name
        self.model_trans, self.input_token_index, self.target_token_index, self.reverse_target_char_index, self.max_encoder_seq_length, self.max_decoder_seq_length, self.num_encoder_tokens, self.num_decoder_tokens = self.load_trans_model(model_name)
    
    
    
    def translate(self, model_opt):
        # (Same as original code for translation)
        # ...
        in_encoder = np.zeros((1, self.max_encoder_seq_length, self.num_encoder_tokens),dtype='float32')

        for t, char in enumerate(model_opt):
            in_encoder[0, t, self.input_token_index[char]] = 1.
        in_encoder[0, t + 1:, self.input_token_index[' ']] = 1.

        in_decoder = np.zeros((len(in_encoder), self.max_decoder_seq_length, self.num_decoder_tokens),dtype='float32')
        in_decoder[:, 0, self.target_token_index["\t"]] = 1

        # 生成 decoder 的 output
        for i in range(self.max_decoder_seq_length - 1):
            predict = self.model_trans.predict([in_encoder, in_decoder])
            predict = predict.argmax(axis=-1)
            predict_ = predict[:, i].ravel().tolist()
            for j, x in enumerate(predict_):
                in_decoder[j, i + 1, x] = 1 # 將每個預測出的 token 設為 decoder 下一個 timestsmp 的輸入

        seq_index = 0
        decoded_sentence = ""
        output_seq = predict[seq_index, :].ravel().tolist()
        for x in output_seq:
            if self.reverse_target_char_index[x] == "\n":
                break
            else:
                decoded_sentence+=self.reverse_target_char_index[x]

        # print('Input sentence:', model_opt)
        # print('Decoded sentence:', decoded_sentence)
        return decoded_sentence
    
    def check_timeout(self):
        # 每過一秒就判斷一次
        # 滿足條件就翻譯語序
        if (self.alarm_set and (abs(time.time() - self.last_updated_time) >= 10)):
            # 修改DATA型態
            data_result = ' '.join(self.data_list)#將[1,2]的資料型態轉為"1 2"
            # 判斷語序
            trans_result = self.translate(data_result)#' '.join(sentence)
            print('---result---', trans_result)
            # 將結果存在trans_list中
            trans_result = urllib.parse.unquote(trans_result)
            # self.trans_list.append(trans_result)
            self.append_to_list(self.trans_list,trans_result)
            
            # 清空 data_list 資料
            self.data_list = []
            #重新計時
            self.last_updated_time = time.time()
            # 將通知取消
            self.alarm_set = False
    

        
    # def add_number(self):
    #     # (Same as original code for add_number)
    #     # ...
    #     num = 0
    #     while True:
    #         time.sleep(5)
    #         self.number_list.append(num)
    #         num += 1
    def switch_hands_model(self, model_name):
        # 終止目前運行的腳本
        hands_model_file = self.available_hands_models[model_name]
        print("切換手語辨識腳本",hands_model_file)
        if self.process:
            self.process.terminate()
            self.process = None

        # 更新當前選擇的模型並重新載入
        self.current_hands_model_name = model_name
        self.process = subprocess.Popen(['python', hands_model_file])


    def stop_external_script(self):
        if self.process:
            self.process.terminate()
            self.process = None
    
    def start_speech(self,speech_name='speech_recognition.py'):
        self.speech_process = subprocess.Popen(['python', speech_name])
    
    def stop_speech(self):
        if self.speech_process:
            self.speech_process.terminate()
            self.speech_process = None
            
    
    def initialize_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html',model_name=self.available_models[self.current_model_name],hands_model_name=self.available_hands_models[self.current_hands_model_name])
        
        @self.app.route('/change_model', methods=['POST'])
        def change_model():
            selected_model = request.form.get('model')
            self.switch_model(selected_model)  # 切換到選擇的模型
            print("切換到語序模型:", selected_model)
            return redirect('/')  # 重定向回主頁
        
        @self.app.route('/change_hands_model', methods=['POST'])
        def change_hands_model():
            selected_hands_model = request.form.get('model')
            self.switch_hands_model(selected_hands_model)  # 切換到選擇的模型
            print(f"切換到手語模型: {selected_hands_model}")
            return redirect('/')
        
        

        @self.app.route('/stop', methods=['POST'])
        def stop():
            self.stop_external_script()
            print("已停止外部腳本。")
            return redirect('/')
        
        @self.app.route('/stopSpeech', methods=['POST'])
        def stopSpeech():
            self.stop_speech()
            print("已停止語音轉文字。")
            return redirect('/')
        @self.app.route('/startSpeech', methods=['POST'])
        def startSpeech():
            self.start_speech()
            print("已開啟語音轉文字。")
            return redirect('/')
        # @self.app.route('/test', methods=['GET'])
        # def get_test():
        #     return jsonify(self.number_list)
        
        @self.app.route('/speech', methods=['POST'])
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
            # self.speech_list.append(speech)
            self.append_to_list(self.speech_list,speech)
            return self.speech_list

        @self.app.route('/speech', methods=['GET'])
        def get_SpeechResult():
            return jsonify(self.speech_list)

        @self.app.route('/RR', methods=['POST'])
        def RecongResult():
            # (Same as original code for /RR POST)
            # ...
            '''
            POST範例
            data = {
                'value': data
                }
            '''
            value = request.form.get('value')
            value = urllib.parse.unquote(value)
            # 將POST資料存進全域變數
            self.data_list.append(value)
            # self.histor_data_list.append(value)
            self.append_to_list(self.histor_data_list,value)
            # 如果有新的手語儲存，就設alarm_set = True，並更新last_updated_time
            # global alarm_set
            self.alarm_set = True
            print("set alarm_set TRue")
            # global last_updated_time
            self.last_updated_time = time.time()
            print(self.last_updated_time,"updated time")
            return self.data_list

        @self.app.route('/RR', methods=['GET'])
        def get_RecongResult():
            return jsonify(self.histor_data_list)

        @self.app.route('/TL', methods=['GET'])
        def get_Translate():
            return jsonify(self.trans_list)
        
        @self.app.route('/DATA', methods=['GET'])
        def get_DATA():
            # history_json = json.dumps(self.histor_data_list)
            # translate_json = json.dumps(self.trans_list)
            # speech_json = json.dumps(self.speech_list)
            # return render_template('data.html', history=history_json, translate=translate_json, speech=speech_json)
            return render_template('data.html', history=self.histor_data_list, translate=self.trans_list, speech=self.speech_list)
        @self.app.route('/get_data', methods=['GET'])
        def get_data():
            return jsonify(history=self.histor_data_list, translate=self.trans_list, speech=self.speech_list)
    def run(self):
        # thread = Thread(target=self.add_number)
        # thread.start()
        self.scheduler.add_job(self.check_timeout, 'interval', seconds=1)
        self.scheduler.start()
        self.app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == '__main__':
    server = Server()
