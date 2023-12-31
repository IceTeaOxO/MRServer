import cv2
import numpy as np
import mediapipe as mp
import os
from mss import mss
from keras.models import load_model
import time
import requests
import threading
from collections import Counter

# 非同步辨識
# 初始化一個全域變數，用於存儲模型預測結果
global prediction_result
prediction_result = None


# 影像串流設定
stream_width = 720#1280  # 影像寬度
stream_height = 440#960  # 影像高度
stream_fps = 60  # 影像串流幀率

# 載入模型
new_model = load_model("./handsModel/model1_service1_0924.keras")#####model1_0730.keras

# actions = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'apply', 'check', 'deposit', 'finish', 'get', 'give_you', 'good',
#                     'i', 'id_card', 'is', 'job', 'money', 'same', 'saving_book', 'sheet', 'sign', 'stamp', 'taiwan', 'take', 'ten_thousand',
#                     'thank_you', 'this', 'thousands', 'transfer', 'transfer_in', 'transfer_out', 'want', 'yes', 'you']) 
actions = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'check', 'finish', 'give_you', 'good',
                    'i', 'id_card', 'is','money', 'saving_book', 'sign', 'taiwan', 'take', 'ten_thousand', 'yes'])


sequence = []
sentence = []
predictions = []
threshold = 0.7
session = requests.Session()
def initMss():
    # 初始化 mss
    sct = mss()
    # 設定影像串流視窗大小
    stream_bbox = {'top': 150, 'left': 150, 'width': stream_width, 'height': stream_height}
    return sct,stream_bbox

def initMediapipe():
    ## 設定mediapipe的模型
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    # 初始化 Mediapipe Holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_holistic,mp_drawing,holistic


# 偵測圖片的手勢，將節點結果傳回
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = True  # 將影像設定為可寫
    results = model.process(image_rgb)
    return image_rgb, results

# 只取節點結果的左右手資料
def extract_keypoints_without_face(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# 將結果送到server
# 使用Session發送POST請求，使用Session對象來重用底層的TCP連接
def sendResult(data):
    url = 'http://127.0.0.1:8080/RR'
    try:
        response = session.post(url, data = {
            'value': data
            })
        if response.status_code == 200:
            print('Data sent successfully.')
        else:
            print('Failed to send data.')
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending data: {e}")

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    ) 



sct,stream_bbox = initMss()
mp_holistic,mp_drawing,holistic = initMediapipe()
    
# 開啟串流視窗
stream_window = cv2.namedWindow('Screen Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Screen Stream', stream_width, stream_height)


# 模型預測執行緒
# prediction_thread = threading.Thread(target=model_prediction_thread, args=(sequence,))
# prediction_thread.start()



while True:
    
    # time.sleep(0.03)
    # 擷取螢幕畫面
    sct_img = sct.grab(stream_bbox)
    frame = np.array(sct_img)

    # 將螢幕畫面進行骨架檢測
    frame, results = mediapipe_detection(frame, holistic)

    # 繪製骨架
    # mp_drawing.draw_landmarks(
    #     frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #     mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
    #     mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    # )
    draw_styled_landmarks(frame, results)

    
        
    ## 預測手語，只取左右手節點
    keypoints = extract_keypoints_without_face(results)
    # 節點數大於20就紀錄為一個frame並加入sequence
    # sequence大小限制為30
    if np.count_nonzero(keypoints) > 30:
        sequence.append(keypoints)
        sequence = sequence[-30:]
        # print(sequence)
    
    
    
    # 進行辨識
    if len(sequence) == 30:
        # 非同步call predict
        
        
        # 這裡是預測結果的機率表，機率<1
        res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
        # 將預測數值最大的結果放入predictions，放入值為index
        # if res[np.argmax(res)] > threshold: 
        #     predictions.append(np.argmax(res))
        predictions.append(np.argmax(res))
        
        # 限制prediction的大小為10
        
        predictions = predictions[-10:]
        # print(predictions)
        
        # 判斷是否有足夠的信心判斷這個手語
        # 為甚麼他是取預測的最小值index相等於當下預測的機率最高的index
        # 他應該是要去重，然後依照出現次數進行排序，反正這邊再跟模型組討論
        
        if np.unique(predictions[-10:])[0]==np.argmax(res):                 
            # 如果機率>0.7
            if res[np.argmax(res)] > threshold: 
                print(actions[np.argmax(res)])
                
                # 印出目前的句子長度
                print(len(sentence))
        # if Counter(predictions[-10:]).most_common(1)[0][0]==np.argmax(res):        
                
                # 卡住???
                if len(sentence) > 0: 
                    # 如果辨識結果不一樣，則加入句子
                    # sentence.append(actions[np.argmax(res)])
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                        sendResult(actions[np.argmax(res)])
                        # 等待伺服器回應可能造成堵塞
                
                else:
                    # 如果句子長度=0，直接將結果加入句子
                    sentence.append(actions[np.argmax(res)])
                    sendResult(actions[np.argmax(res)])
                    # print(sentence)
                    # 這裡是將手語組成片段句子的邏輯，看情況使用
                print(sentence)
        if len(sentence) > 5: 
            # 只記錄最近的五個詞語
            sentence = sentence[-5:]
                
        # 將圖像轉換為 BGR 格式
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
    # 顯示影像
    cv2.imshow('Screen Stream', frame_bgr)
        
        
                
    # print(sentence)
    # 若按下 'q' 鍵則結束迴圈
    if cv2.waitKey(1) == ord('q'):
        break

# 關閉串流視窗
cv2.destroyAllWindows()