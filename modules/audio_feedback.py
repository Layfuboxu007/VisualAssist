import pyttsx3
import threading
import time
import queue

class AudioFeedback:
    def __init__(self, rate=150, volume=1.0, cooldown=15.0,
                 critical_cooldown=15.0, close_cooldown=20.0,
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
        
        # Thread-safe PriorityQueue for background speech processing
        self.speech_queue = queue.PriorityQueue()
        self.last_speech_end_time = 0.0
        self.is_speaking = False
        
        # Start persistent speech background worker thread
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._speech_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _speech_worker(self):
        """Single background worker thread that processes alerts sequentially.
        COM and pyttsx3 are initialized and cleaned up fresh for each text to guarantee audio output on Windows.
        """
        while self.is_running:
            try:
                try:
                    # Non-blocking pull with timeout to allow thread exit checks
                    priority, ts, text, done_event = self.speech_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if text:
                    try:
                        self.is_speaking = True
                        
                        # Re-initialize COM on this thread for this execution
                        import ctypes
                        ctypes.windll.ole32.CoInitialize(None)
                        
                        # Re-initialize the pyttsx3 engine fresh
                        engine = pyttsx3.init()
                        engine.setProperty('rate', self.rate)
                        engine.setProperty('volume', self.volume)
                        self.current_engine = engine
                        
                        engine.say(text)
                        engine.runAndWait()
                        
                        # Release engine resources
                        self.current_engine = None
                        del engine
                        
                        # Uninitialize COM
                        ctypes.windll.ole32.CoUninitialize()
                    except Exception as e:
                        print(f"[AUDIO ERROR] Speech execution failed: {e}")
                    finally:
                        self.is_speaking = False
                
                self.last_speech_end_time = time.time()
                
                # Signal completion if this is a blocking/synchronous query
                if done_event:
                    done_event.set()
                    
                self.speech_queue.task_done()
                
            except Exception as e:
                print(f"[AUDIO WORKER ERROR] Background loop encountered exception: {e}")

    def speak_sync(self, text):
        """Speak synchronously (blocks execution until finished)."""
        print(f"[AUDIO OUT] {text}")
        
        # Stop current engine speech immediately to prevent lingering SAPI5 audio
        if hasattr(self, "current_engine") and self.current_engine:
            try:
                self.current_engine.stop()
            except Exception:
                pass
            self.current_engine = None
            
        # Clear queue first to avoid waiting for stale alerts
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except (queue.Empty, ValueError):
                break
                
        done_event = threading.Event()
        # Priority 0 is highest, time.time() acts as FIFO tie-breaker
        self.speech_queue.put((0, time.time(), text, done_event))
        done_event.wait()

    def speak(self, text, is_critical=False):
        """Speak asynchronously by adding text to priority queue.
        Enforces global silence interval and word cooldowns to prevent spam.
        """
        current_time = time.time()
        
        # If currently speaking, reject non-critical alerts to prevent queue backlog
        if self.is_speaking and not is_critical:
            return False
            
        # Enforce global silence interval between consecutive spoken alerts (bypass for critical alerts)
        if not is_critical:
            silence_interval = self.global_silence_interval
            if current_time - self.last_speech_end_time < silence_interval:
                return False  # Drop, within silence window
        
        # Check cooldown for this specific alert text
        # If critical, we still check a smaller safety cooldown (6s) to prevent self-interruption and stutter
        safety_cooldown = 6.0 if is_critical else self.cooldown
        if text in self.last_spoken:
            if current_time - self.last_spoken[text] < safety_cooldown:
                return False  # Skip, within cooldown window
        
        self.last_spoken[text] = current_time
        print(f"[AUDIO OUT (async)] {text}")
        
        # If critical, interrupt any current speech immediately
        if is_critical and hasattr(self, "current_engine") and self.current_engine:
            try:
                self.current_engine.stop()
            except Exception:
                pass
            self.current_engine = None

        # Clear the queue to discard any older/stale pending speech requests
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except (queue.Empty, ValueError):
                break
                
        priority = 0 if is_critical else 1
        self.speech_queue.put((priority, time.time(), text, None))
        return True

    def speak_obstacles(self, sorted_obstacles, close_threshold, critical_threshold):
        """Processes all detected obstacles in priority order, but speaks at most ONE alert
        that is not currently on cooldown, avoiding concurrent alerts fighting/stuttering.
        """
        current_time = time.time()
        
        # 1. Clean up stale obstacles (not seen for > 2.0 seconds)
        for key in list(self.tracked_obstacles.keys()):
            if current_time - self.tracked_obstacles[key]['last_seen'] > 2.0:
                del self.tracked_obstacles[key]

        # 2. Iterate in priority order and find the first candidate to speak
        for obs in sorted_obstacles:
            label = obs["label"]
            position = obs["position"]
            distance = obs.get("distance", 99.0)
            
            # Classify threat zone
            if distance < critical_threshold:
                current_zone = "critical"
            elif distance < close_threshold:
                current_zone = "close"
            else:
                current_zone = "far"
                
            if current_zone == "far":
                continue  # Ignore far obstacles
                
            key = (label, position)
            
            # Initialize tracking if new
            if key not in self.tracked_obstacles:
                self.tracked_obstacles[key] = {
                    'last_seen': current_time,
                    'last_alert_time': 0.0,
                    'last_alert_zone': 'far'
                }
            else:
                self.tracked_obstacles[key]['last_seen'] = current_time
                
            info = self.tracked_obstacles[key]
            
            # Check if we should alert for this obstacle based on state
            should_alert = False
            urgency = {"far": 0, "close": 1, "critical": 2}
            
            if urgency[current_zone] > urgency[info['last_alert_zone']]:
                # Zone upgrade: immediate alert
                should_alert = True
            else:
                zone_cooldown = self.critical_cooldown if current_zone == "critical" else self.close_cooldown
                if current_time - info['last_alert_time'] >= zone_cooldown:
                    should_alert = True
                    
            if should_alert:
                # Format phrase
                if position == "ahead":
                    alert_phrase = f"{label.capitalize()} ahead"
                else:
                    alert_phrase = f"{label.capitalize()} {position}"
                    
                is_critical = (current_zone == "critical")
                
                # Try speaking it
                if self.speak(alert_phrase, is_critical=is_critical):
                    # Successfully spoke (or queued) -> update state and EXIT.
                    # This ensures we speak AT MOST ONE alert per frame loop!
                    info['last_alert_time'] = current_time
                    info['last_alert_zone'] = current_zone
                    break  # Stop processing further obstacles for this frame!
                else:
                    # Speak was throttled or rejected due to global checks
                    pass
            else:
                # We didn't speak it because of cooldowns, but we still track it as seen.
                # (We don't break the loop, because we want to see if a lower-priority
                # obstacle that is NOT on cooldown can be spoken instead!)
                pass

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
        print(f"[ALARM] Triggering alarm: {alarm_type}")
        self.speak(f"Warning! {alarm_type.replace('_', ' ')} detected!", is_critical=True)

    def speak_navigation(self, steer_advice):
        """Provide directional navigation warnings based on the steer advice."""
        if steer_advice == "clear":
            return
            
        current_time = time.time()
        # Enforce a 12.0 second cooldown for the same steering direction to avoid spamming
        last_nav_time = getattr(self, "_last_nav_time", 0.0)
        last_nav_advice = getattr(self, "_last_nav_advice", "")
        
        if current_time - last_nav_time < 12.0 and steer_advice == last_nav_advice:
            return
            
        self._last_nav_time = current_time
        self._last_nav_advice = steer_advice
        
        if steer_advice == "steer left":
            self.speak("Obstacle ahead. Steer left.", is_critical=True)
        elif steer_advice == "steer right":
            self.speak("Obstacle ahead. Steer right.", is_critical=True)
        elif steer_advice == "stop":
            self.speak("Path blocked. Stop.", is_critical=True)

    def release(self):
        """Stops the background worker thread and clears the queue."""
        self.is_running = False
        
        # Stop current engine speech immediately to prevent SAPI5 lingering audio
        if hasattr(self, "current_engine") and self.current_engine:
            try:
                self.current_engine.stop()
            except Exception:
                pass
            self.current_engine = None
            
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except (queue.Empty, ValueError):
                break
