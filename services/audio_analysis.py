import speech_recognition as sr
import threading

class SpeechRecognition:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.speech = ""
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start_listening(self):
        """Start background listening for speech."""
        def listen():
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("[AudioRecognition] Listening started...")
                while self.running:
                    try:
                        audio = self.recognizer.listen(source, timeout=5)
                        text = self.recognizer.recognize_google(audio)
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
    
    def analyse_audio(): #FOR EMOTION ANALYSIS PURPOSES
        pass