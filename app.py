import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, app, render_template, Response, jsonify
from facial_recognition import FacialRecognition
import threading
import time, asyncio


app = Flask(__name__)

facial_recognition = FacialRecognition()
current_emotion = "neutral"
emotion_lock = threading.Lock()

async def update_emotion():
    global current_emotion
    while True:
        detected_emotion = await facial_recognition.getEmotion()
        if detected_emotion:
            with emotion_lock:
                current_emotion = detected_emotion
        await asyncio.sleep(1)

def start_async_emotion_fetch():
    asyncio.run(update_emotion())

threading.Thread(target=start_async_emotion_fetch, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def getVideo():
        while True:
            frame = facial_recognition.getFrame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(getVideo(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion-fetch')
def emotion():
    global current_emotion
    with emotion_lock:
        return jsonify({'emotion': current_emotion})


if __name__ == '__main__':
    app.run(debug=True)