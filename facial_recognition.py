import cv2
from deepface import DeepFace
import threading, asyncio
import concurrent.futures
from collections import deque

executor = concurrent.futures.ThreadPoolExecutor()

class FacialRecognition:
    def __init__(self):
        self.image = cv2.VideoCapture(0)#, cv2.CAP_DSHOW)
        self.height = int(self.image.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.image.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.current_emotion = "neutral"
        self.lock = threading.Lock()
        self.last_frame = None

    def analyze_emotion_sync(self, frame):
        return DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)[0]['dominant_emotion']

    async def analyze_emotion(self, frame):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.analyze_emotion_sync, frame)

    '''async def analyze_emotion(self, frame):
        analysis = DeepFace.analyze(frame, actions=['emotion'])
        return analysis[0]['dominant_emotion'] if analysis is not None else "neutral"'''
    
    async def getEmotion(self):
        with self.lock:
            if self.last_frame is not None:
                self.current_emotion = await self.analyze_emotion(self.last_frame)
                return self.current_emotion
        return "neutral"
        
    def getFrame(self):
        ret, frame = self.image.read()
        if not ret:
            return None

        r_frame = cv2.resize(frame, (int(self.width * 0.6), int(self.height * 0.6)))

        with self.lock:
            self.last_frame = frame
        
        _, jpeg = cv2.imencode('.jpeg', r_frame)
            
        return jpeg.tobytes() if jpeg is not None else None

    '''async def getFrame(self):
        with self.lock:
            ret, frame = self.image.read()
            if not ret:
                return None, "neutral"

            frame = cv2.resize(frame, (int(self.width * 0.6), int(self.height * 0.6)))
            
            _, jpeg = cv2.imencode('.jpeg', frame)
            if jpeg is None:
                return None, "neutral"

            current_emotion = await self.analyze_emotion(frame)

            return jpeg.tobytes(), current_emotion'''

'''import cv2
from deepface import DeepFace
import numpy as np


class FacialRecognition():
    def __init__(self):
        self.image = cv2.VideoCapture(0)
        self.height = int(self.image.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.image.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.current_emotion = "neutral"

    def getFrame(self):
        ret, frame = self.image.read()

        if not ret:
            return None, None
        
        frame = cv2.resize(frame, (int(self.width * 0.6), int(self.height * 0.6)))
        
        _, jpeg = cv2.imencode('.jpg', frame)

        if jpeg is None:
            return None, None

        nparr = np.frombuffer(jpeg.tobytes(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        current_emotion = self.getEmotion(img)

        return jpeg.tobytes(), current_emotion
    
    def getEmotion(self, img):
        try:
            analysis = DeepFace.analyze(img_path=img, actions=['emotion'])
            return analysis[0]['dominant_emotion']
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return None'''