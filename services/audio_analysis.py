import speech_recognition as sr
from transformers import pipeline
import librosa
import numpy as np
import threading
from tf_keras import models

class SpeechRecognition:
    def __init__(self):
        self.model = models.load_model('./models/emotion_model.h5')
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.speech = ""
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.audio = None
        self.emotion = 'neutral'

    def _update_speech(self, text):
        with self.lock:
            self.speech = text
            print(f"[AudioRecognition] Recognized: {text}")

    def start_listening(self):
        """Start background listening for speech."""
        def listen():
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("[AudioRecognition] Listening started...")
                while self.running:
                    try:
                        self.audio = self.recognizer.listen(source, timeout=5)
                        text = self.recognizer.recognize_google(self.audio, language='es-ES')
                        self._update_speech(text)
                    except sr.UnknownValueError:
                        print("[AudioRecognition] Could not understand audio.")
                    except sr.RequestError as e:
                        print(f"[AudioRecognition] Recognition error: {e}")
                    except Exception as e:
                        print(f"[AudioRecognition] Error: {e}")

        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=listen)
            self.thread.daemon = True
            self.thread.start()

    def stop_listening(self):
        """Stop background listening."""
        if self.running:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join()
            print("[AudioRecognition] Listening stopped.")
            
        else:
            print("[AudioRecognition] Listening is not active.")

    def get_recognized_text(self):
        """Get the latest recognized text."""
        with self.lock:
            if self.speech != "":
                yield self.speech

    def get_emotion(self):
        features = self._extract_features()
        features = np.expand_dims(features, axis=0)
        pred = self.model.predict(features)
        emotion = np.argmax(pred)
        self.emotion = str(emotion)

    def _extract_features(self):
        audio_array = np.frombuffer(self.audio.get_wav_data(), dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / np.max(np.abs(audio_array))

        mfcc = librosa.feature.mfcc(y=audio_array, sr=22050, n_mfcc=59)
        return np.mean(mfcc.T, axis=0)