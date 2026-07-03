import pyttsx3
import threading
import time

class AudioFeedback:
    def __init__(self, rate=150, volume=1.0, cooldown=2.0,
                 critical_cooldown=2.0, close_cooldown=4.0,
                 distance_update_threshold=0.5,
                 global_silence_interval=2.5,
                 critical_silence_interval=1.0):
        self.rate = rate
        self.volume = volume
        self.cooldown = cooldown
        
        # Cooldown tracking parameters
        self.critical_cooldown = critical_cooldown
        self.close_cooldown = close_cooldown
        
        # Silence interval limits
        self.global_silence_interval = global_silence_interval
        self.critical_silence_interval = critical_silence_interval
        
        self.last_spoken = {}  # maps text to timestamp of when it was last spoken
        self.tracked_obstacles = {}  # key: (label, position) -> dict of status
        
        # Thread safety & global silence tracking
        self._lock = threading.Lock()
        self._is_speaking = False
        self.last_speech_end_time = 0.0

    def _speak_thread(self, text):
        """Internal method to run speech generation in a temporary thread with COM initialization."""
        try:
            # Initialize COM library for SAPI5 SndPlaySound on Windows
            import ctypes
            ctypes.windll.ole32.CoInitialize(None)
        except Exception:
            pass

        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[AUDIO ERROR] Speech failed: {e}")
        finally:
            try:
                import ctypes
                ctypes.windll.ole32.CoUninitialize()
            except Exception:
                pass
            with self._lock:
                self._is_speaking = False
            self.last_speech_end_time = time.time()

    def speak_sync(self, text):
        """Speak synchronously (blocks execution until finished)."""
        print(f"[AUDIO OUT] {text}")
        with self._lock:
            self._is_speaking = True
        t = threading.Thread(target=self._speak_thread, args=(text,))
        t.daemon = True
        t.start()
        t.join()

    def speak(self, text, is_critical=False):
        """Speak asynchronously by spawning a temporary thread.
        Enforces a global silence interval to prevent audio spam.
        """
        current_time = time.time()
        
        # Enforce global silence interval between consecutive spoken alerts
        silence_interval = self.critical_silence_interval if is_critical else self.global_silence_interval
        if current_time - self.last_speech_end_time < silence_interval:
            return False  # Drop, within silence window
        
        # Check cooldown for this specific alert text
        if text in self.last_spoken:
            if current_time - self.last_spoken[text] < self.cooldown:
                return False  # Skip, within cooldown window
        
        with self._lock:
            if self._is_speaking:
                return False  # Drop if already speaking
            self._is_speaking = True

        self.last_spoken[text] = current_time
        print(f"[AUDIO OUT (async)] {text}")
        t = threading.Thread(target=self._speak_thread, args=(text,))
        t.daemon = True
        t.start()
        return True

    def speak_obstacle(self, label, position, distance, close_threshold, critical_threshold):
        """Speak alert for a specific obstacle with smart state-based tracking and cooldowns."""
        current_time = time.time()
        
        # Clean up stale obstacles (not seen for > 2.0 seconds)
        for key in list(self.tracked_obstacles.keys()):
            if current_time - self.tracked_obstacles[key]['last_seen'] > 2.0:
                del self.tracked_obstacles[key]

        # Classify the threat level zone
        if distance < critical_threshold:
            current_zone = "critical"
        elif distance < close_threshold:
            current_zone = "close"
        else:
            current_zone = "far"

        # Unique identifier for the obstacle
        key = (label, position)

        # Initialize tracking if new
        if key not in self.tracked_obstacles:
            self.tracked_obstacles[key] = {
                'last_seen': current_time,
                'last_alert_time': 0.0,
                'last_alert_zone': 'far'
            }
        else:
            # Update last seen timestamp
            self.tracked_obstacles[key]['last_seen'] = current_time

        info = self.tracked_obstacles[key]

        # Determine whether we should alert
        should_alert = False

        # Only alert for close and critical obstacles; silent for far
        if current_zone != "far":
            # Map zone names to integer urgency levels for comparison
            urgency = {"far": 0, "close": 1, "critical": 2}
            
            # Transition to a more urgent zone triggers an immediate alert
            if urgency[current_zone] > urgency[info['last_alert_zone']]:
                should_alert = True
            # Same or lower zone checks the standard cooldown timers
            else:
                zone_cooldown = self.critical_cooldown if current_zone == "critical" else self.close_cooldown
                if current_time - info['last_alert_time'] >= zone_cooldown:
                    should_alert = True
        else:
            # If it went far, update the alert zone to "far" to track that it is no longer close
            info['last_alert_zone'] = "far"

        if should_alert:
            # Format the verbal alert phrase simply as "[Label] [Position]" (e.g. "Person ahead")
            if position == "ahead":
                alert_phrase = f"{label.capitalize()} ahead"
            else:
                alert_phrase = f"{label.capitalize()} {position}"

            # Attempt to speak the phrase asynchronously
            is_critical = (current_zone == "critical")
            if self.speak(alert_phrase, is_critical=is_critical):
                # Only update alert state if speech was successfully triggered (not dropped/busy)
                info['last_alert_time'] = current_time
                info['last_alert_zone'] = current_zone

    def play_alarm(self, alarm_type="generic"):
        """Plays a warning sound/alarm."""
        # TODO: Implement actual alarm files playing via pygame or winsound
        print(f"[ALARM] Triggering alarm: {alarm_type}")
        self.speak(f"Warning! {alarm_type.replace('_', ' ')} detected!")
