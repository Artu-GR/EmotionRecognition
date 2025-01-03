import cv2
from deepface import DeepFace
import threading, asyncio
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor()

class FacialRecognition:
    def __init__(self):
        self.image = cv2.VideoCapture(0)
        self.height = int(self.image.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.image.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        self.current_emotion = "neutral"
        self.lock = threading.Lock()
        self.last_frame = None
        self._start_emotion_thread()

    def analyze_emotion_sync(self, frame):
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            return analysis[0]['dominant_emotion']
        except Exception as e:
            print(f"[FacialRecognition] Analysis Error: {e}")
            return "neutral"

    async def analyze_emotion(self, frame):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.analyze_emotion_sync, frame)
    
    async def getEmotion(self):
        with self.lock:
            if self.last_frame is not None:
                self.current_emotion = await self.analyze_emotion(self.last_frame)
                return self.current_emotion
        return "neutral"
        
    def getFrame(self):
        ret, frame = self.image.read()
        if not ret:
            print("[FacialRecognition] Failed to capture video frame.")
            return None

        r_frame = cv2.resize(frame, (int(self.width * 0.6), int(self.height * 0.6)))

        with self.lock:
            self.last_frame = frame
        
        _, jpeg = cv2.imencode('.jpeg', r_frame)
            
        return jpeg.tobytes() if jpeg is not None else None
    
    def release(self):
        self.image.release()
        print("[FacialRecognition] Video capture released.")
        
    async def update_emotion(self):
        while True:
            await self.getEmotion()
            await asyncio.sleep(1)

    def _start_emotion_thread(self):
        threading.Thread(target=self.run_emotion_loop, daemon=True).start()

    def run_emotion_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.update_emotion())