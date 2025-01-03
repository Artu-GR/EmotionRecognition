import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, app, render_template, Response, jsonify
from services.facial_recognition import FacialRecognition
from services.audio_analysis import SpeechRecognition
import threading
import asyncio


app = Flask(__name__)

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

@app.route('/audio_analysis', methods=['POST'])
def audio_analysis():
    Audio.start_listening()
    speech = Audio.get_recognized_text()
    return {"speech":speech}

@app.route('/stop-mic')
def stop_mic():
    Audio.stop_listening()
    return jsonify({"message":"Microphon stopped"})

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