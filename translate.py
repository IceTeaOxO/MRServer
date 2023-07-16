import os
import numpy as np
import time
from tensorflow.keras.models import load_model

alarm_set = True

def get_dataset(data_path_trans):
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
    return input_texts,target_texts,input_characters,target_characters

data_path_trans = 'EngToChinese.txt'
input_texts,target_texts,input_characters,target_characters = get_dataset(data_path_trans)

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
    

model_trans = load_model("model0529-20.h5")

def translate(model_opt='check yes paper sign'):
    # model_opt = 'check yes paper sign'
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

last_updated_time=0
current_time = time.time()  
if alarm_set and current_time - last_updated_time >= 10:
            # 時間過10秒，將 sentence 放入下一個模型進行預測
            trans_result = translate()#' '.join(sentence)
            print('---result---', trans_result)
            # 清空 sentence 資料
            alarm_set = False
            sequence = []
