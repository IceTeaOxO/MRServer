import os
import azure.cognitiveservices.speech as speechsdk
import keyboard
import requests

def recognize_from_microphone():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    # speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    speech_config = speechsdk.SpeechConfig(subscription='e9ea122e90f94c999d884ce0c240c77d', region='eastasia')
    speech_config.speech_recognition_language="zh-TW"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    # speech_recognition_result = speech_recognizer.recognize_once_async().get()
    while True:
        speech_recognition_result = speech_recognizer.recognize_once_async().get()
        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(speech_recognition_result.text))
            
            # 將辨識結果送到伺服器
            # speech_recognition_result.text
            encoded_text = speech_recognition_result.text.encode('utf-8')
            data = {
                'speech': encoded_text
            }
            url = 'http://127.0.0.1:8080/speech'
            response = requests.post(url, data=data)
            print("Response from server: {}".format(response.text))
            
        elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_recognition_result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")
        if keyboard.is_pressed('q'):
            break
recognize_from_microphone()