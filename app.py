import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, app, render_template, Response, jsonify
from services.facial_recognition import FacialRecognition
from services.audio_analysis import SpeechRecognition
import time

emotion_dict = {
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fear',
    7: 'disgust',
    8: 'surprised'
}

app = Flask(__name__)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

Video = FacialRecognition()
Audio = SpeechRecognition()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('facial_recognition.html')

@app.route('/audio')
def audio():
    return render_template('audio_analysis.html')

@app.route('/audio_analysis')#, methods=['POST'])
def audio_analysis():

    def generate_audio():
        Audio.start_listening()
        last_speech = None

        try:
            while True:
                speech = next(Audio.get_recognized_text(), None)
                if speech and speech != last_speech:
                    last_speech = speech
                    yield f"data: {speech}\n\n"  # Send it as a stream
                time.sleep(0.5)
        except GeneratorExit:
            Audio.stop_listening()
    
    return Response(generate_audio(), mimetype='text/event-stream')

@app.route('/stop-mic')
def stop_mic():
    Audio.stop_listening()
    return jsonify({"message":"Microphone stopped"})

@app.route('/get-speech-emotion')    
def get_speech_emotion():
    Audio.get_emotion()
    if Audio.emotion is not None:
        return jsonify({'emotion': emotion_dict[int(Audio.emotion)]})
    else:
        return jsonify({'message': "Emotion analysis not available."}), 400

@app.route('/video_feed')
def video_feed():
    def getVideo():
        while True:
            frame = Video.getFrame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(getVideo(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion-fetch')
def emotion():
    return jsonify({'emotion': Video.current_emotion})

@app.route('/release_camera', methods=['POST'])
def release_camera():
    Video.release()


if __name__ == '__main__':
    app.run(debug=True)