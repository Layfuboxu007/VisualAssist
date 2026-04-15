import pyttsx3
import threading

class AudioFeedback:
    def __init__(self, rate=150, volume=1.0):
        # Initialize pyttsx3. We run the initialization on the main thread.
        # But we will use threading to speak without blocking vision loops.
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self._is_speaking = False

    def speak_sync(self, text):
        """Speak synchronously (blocks execution until finished)."""
        print(f"[AUDIO OUT] {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def _speak_thread(self, text):
        """Internal method to run speech generation in a thread."""
        self._is_speaking = True
        try:
            # We initialize a new pyttsx3 instance per thread to avoid COM errors on Windows
            thread_engine = pyttsx3.init()
            thread_engine.setProperty('rate', self.engine.getProperty('rate'))
            thread_engine.setProperty('volume', self.engine.getProperty('volume'))
            thread_engine.say(text)
            thread_engine.runAndWait()
        finally:
            self._is_speaking = False

    def speak(self, text):
        """Speak asynchronously to avoid blocking the main vision loop."""
        if self._is_speaking:
            # Drop the phrase or queue it if already speaking
            pass 
        else:
            print(f"[AUDIO OUT (async)] {text}")
            t = threading.Thread(target=self._speak_thread, args=(text,))
            t.daemon = True
            t.start()

    def play_alarm(self, alarm_type="generic"):
        """Plays a warning sound/alarm."""
        # TODO: Implement actual alarm files playing via pygame or winsound
        print(f"[ALARM] Triggering alarm: {alarm_type}")
        self.speak(f"Warning! {alarm_type.replace('_', ' ')} detected!")
