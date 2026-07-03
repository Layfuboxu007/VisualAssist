import os
import cv2
import time
from datetime import datetime

class EmergencyModule:
    def __init__(self, output_dir="emergency_alerts"):
        self.output_dir = output_dir
        self.active = False
        self.last_trigger_time = 0.0
        self.cooldown = 10.0  # seconds between automatic fall triggers
        
        # Camera abnormalities state
        self.prev_gray = None
        self.black_start = None
        self.static_start = None
        self.black_duration = 8.0  # seconds to trigger on black screen
        self.static_duration = 10.0 # seconds to trigger on stationary screen (device dropped)
        
        # Ensure emergency alerts directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def trigger_emergency(self, frame, reason="Manual Alert", audio=None, logger=None):
        """
        Triggers emergency state, saves the visual proof, writes logs,
        and provides immediate verbal/alarm warnings.
        """
        current_time = time.time()
        self.active = True
        self.last_trigger_time = current_time

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save frame as visual evidence of emergency
        frame_filename = f"emergency_frame_{timestamp_str}.jpg"
        frame_path = os.path.join(self.output_dir, frame_filename)
        
        # Draw visual markers on the saved frame
        marked_frame = frame.copy()
        h, w = marked_frame.shape[:2]
        cv2.rectangle(marked_frame, (0, 0), (w, h), (0, 0, 255), 15)
        cv2.putText(marked_frame, "EMERGENCY ALERT SYSTEM ACTIVE", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(marked_frame, f"Reason: {reason}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(marked_frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imwrite(frame_path, marked_frame)
        print(f"[EMERGENCY] Saved distress frame to {frame_path}")

        # 2. Write emergency text log file (with simulated GPS coordinates)
        log_filename = f"emergency_log_{timestamp_str}.txt"
        log_path = os.path.join(self.output_dir, log_filename)
        
        gps_lat = "14.5995 N"
        gps_lon = "120.9842 E"
        
        log_content = (
            "===========================================\n"
            "         SIGHTASSIST EMERGENCY LOG          \n"
            "===========================================\n"
            f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Trigger Type : {reason}\n"
            f"GPS Location : Latitude {gps_lat}, Longitude {gps_lon} (Mock Coordinates)\n"
            f"Camera Frame : {frame_filename}\n"
            "Status       : Caregiver notified via simulated SMS and Email alert protocols.\n"
            "-------------------------------------------\n"
        )
        
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(log_content)
            print(f"[EMERGENCY] Saved distress report log to {log_path}")
        except Exception as e:
            print(f"[EMERGENCY ERROR] Failed to write log: {e}")

        # 3. Trigger events & audio alerts
        if logger:
            logger.log_event("emergency", {
                "reason": reason,
                "gps": f"{gps_lat}, {gps_lon}",
                "log_file": log_path,
                "frame_file": frame_path
            })

        if audio:
            audio.play_alarm(reason)
            audio.speak("Emergency alert triggered. caregiver notified.")

    def clear_emergency(self, audio=None):
        """Clears the active emergency state."""
        if self.active:
            self.active = False
            print("[EMERGENCY] Emergency state cleared by user.")
            if audio:
                audio.speak("Emergency cleared. returning to normal operation.")

    def check_fall_detection(self, detected_obstacles, frame, audio=None, logger=None):
        """
        Analyzes detected obstacles to identify a user falling.
        Returns:
            bool: True if emergency was triggered by fall detection, False otherwise.
        """
        if self.active:
            return False

        current_time = time.time()
        if current_time - self.last_trigger_time < self.cooldown:
            return False

        for obs in detected_obstacles:
            # Fall detection heuristic:
            # If the user's camera captures a person lying horizontally on the ground plane,
            # indicating a potential collapse or fall.
            if obs["label"] == "person":
                x1, y1, x2, y2 = obs["box"]
                wb = x2 - x1
                hb = y2 - y1
                
                # A person lying down has a horizontal aspect ratio
                if hb > 0:
                    ratio = wb / hb
                    # Strict fall detection: aspect ratio > 2.2 and bottom of bounding box is low on the ground plane
                    if ratio > 2.2 and y2 > (frame.shape[0] * 0.8):
                        reason = "Fall Detected"
                        print(f"[EMERGENCY] Fall detected based on aspect ratio: {ratio:.2f}")
                        self.trigger_emergency(frame, reason=reason, audio=audio, logger=logger)
                        return True
                        
        return False

    def check_camera_emergency(self, frame, obstacles, audio=None, logger=None):
        """
        Monitors the camera feed for abnormal conditions:
        1. Black Screen (average intensity < 8.0) for 8+ seconds.
        2. Stationary Screen (motion difference < 0.5% pixels) AND no objects detected
           (len(obstacles) == 0, e.g. device is on the floor/ceiling) for 10+ seconds.
        """
        if self.active:
            self.black_start = None
            self.static_start = None
            return False

        current_time = time.time()
        # Fallback check for offline/cooldown constraints
        if current_time - self.last_trigger_time < self.cooldown:
            return False

        import numpy as np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Check for Black Screen / Covered Camera
        mean_brightness = np.mean(gray)
        if mean_brightness < 8.0:
            if self.black_start is None:
                self.black_start = current_time
            elif current_time - self.black_start >= self.black_duration:
                reason = "Camera Obscured Alert"
                print(f"[EMERGENCY] Camera feed is black/covered (brightness: {mean_brightness:.1f})")
                self.trigger_emergency(frame, reason=reason, audio=audio, logger=logger)
                self.black_start = None
                return True
        else:
            self.black_start = None

        # 2. Check for Stationary / Frozen / No Motion Screen AND No Objects Detected
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            # Absolute difference
            diff = cv2.absdiff(self.prev_gray, gray)
            _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            changed_pixels = np.sum(thresh == 255)
            total_pixels = gray.size
            change_ratio = changed_pixels / total_pixels
            
            # Less than 0.5% pixels changed -> stationary
            if change_ratio < 0.005:
                # If also no objects are detected, indicating the device is likely lying flat on the floor/ceiling
                if len(obstacles) == 0:
                    if self.static_start is None:
                        self.static_start = current_time
                    elif current_time - self.static_start >= self.static_duration:
                        reason = "Device Dropped / User Collapse Alert"
                        print(f"[EMERGENCY] Camera feed is stationary and no objects detected (device dropped/floor)")
                        self.trigger_emergency(frame, reason=reason, audio=audio, logger=logger)
                        self.static_start = None
                        return True
                else:
                    self.static_start = None
            else:
                self.static_start = None
        else:
            self.static_start = None
            
        self.prev_gray = gray.copy()
        return False
