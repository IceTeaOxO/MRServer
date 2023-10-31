import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.models import load_model


import cv2
import mediapipe as mp
from collections import Counter
actions = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'check', 'finish', 'give_you', 'good',
                    'i', 'id_card', 'is', 'money', 'saving_book', 'sign', 'taiwan', 'take', 'ten_thousand','yes']) 

new_model = load_model('handsModel/model1_service1_0924.keras')#


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
colors = [(245,117,16)] * 24
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
        cv2.rectangle(output_frame, (0,60+num*17), (int(prob*100), 90+num*17), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*17), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame
def mediapipe_detection(image, model):
    # Transfer image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # Make prediction
    results = model.process(image)
    return results
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
def extract_keypoints_without_face(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh]) 

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.7
alarm_set = False
trans_result =""

cap = cv2.VideoCapture("C:/Users/sandy/Downloads/1696204903829.mp4")#0#C:/Users/sandy/Downloads/1696204903829.mp4#C:/Users/sandy/Downloads/1696176653781.mp4
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        results = mediapipe_detection(frame, holistic)
        # Draw landmarks
        draw_styled_landmarks(frame, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints_without_face(results)
        if np.count_nonzero(keypoints) > 30:
            sequence.append(keypoints)
            sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
            if res[np.argmax(res)] > threshold: 
                predictions.append(np.argmax(res))


            
            
        #3. Viz logic
            if Counter(predictions[-10:]).most_common(1)[0][0]==np.argmax(res):      
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                        sequence = []
                        last_updated_time = time.time()
                        alarm_set = True
                else:
                    sentence.append(actions[np.argmax(res)])
                    sequence = []
                    last_updated_time = time.time()
                    alarm_set = True

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            frame = prob_viz(res, actions, frame, colors)

        # current_time = time.time()  
        # if alarm_set and current_time - last_updated_time >= 10:
        #     # 時間過10秒，將 sentence 放入下一個模型進行預測
        #     trans_result = translate(' '.join(sentence))
        #     print('---result---', trans_result)
        #     # 清空 sentence 資料
        #     alarm_set = False
        #     sequence = []
        #     sentence = []
            
        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()