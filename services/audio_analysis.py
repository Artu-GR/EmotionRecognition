import speech_recognition as sr
from transformers import pipeline
import librosa
import numpy as np
import threading

class SpeechRecognition:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.speech = ""
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.audio = None

    def start_listening(self):
        """Start background listening for speech."""
        def listen():
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("[AudioRecognition] Listening started...")
                while self.running:
                    try:
                        self.audio = self.recognizer.listen(source, timeout=5)
                        text = self.recognizer.recognize_google(self.audio)
                        with self.lock:
                            self.speech = text
                        print(f"[AudioRecognition] Recognized: {text}")
                    except sr.UnknownValueError:
                        print("[AudioRecognition] Could not understand audio.")
                    except sr.RequestError as e:
                        print(f"[AudioRecognition] Recognition error: {e}")
                    except Exception as e:
                        print(f"[AudioRecognition] Error: {e}")

        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=listen)
            #self.thread.daemon = True
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
            return self.speech
        
    # def analyze_speech(self):
    #     speech_emotion = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    #     text_result = speech_emotion(self.speech)
    #     print("Emotion: ", text_result[0]['label'])
    #     return text_result[0]['label'], text_result[0]['score']

    # def audio_features(self): #FOR EMOTION ANALYSIS PURPOSES
    #     mfcc = librosa.feature.mfcc(y=self.audio, n_mfcc = 13)
    #     feature_vector = np.mean(mfcc.T, axis=0)
    #     return feature_vector
    
    # def _analyze(self):
    #     speech_emotion, confidence_speech = self.analyze_speech()
