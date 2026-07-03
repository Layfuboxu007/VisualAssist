import cv2
import yaml
import sys
import time
import os
import tkinter as tk
from tkinter import simpledialog
import threading

from modules.camera import CameraModule
from modules.audio_feedback import AudioFeedback
from modules.vision_yolo import YoloVision
from modules.color_detect import ColorIdentifier
from modules.text_reader import TextReader
from modules.face_id import FaceRecognizer
from modules.emergency import EmergencyModule
from modules.report_logger import ReportLogger

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Could not load config: {e}")
        return None

def describe_scene_async(frame, obstacles, detected_faces, audio, text_reader, logger):
    """
    Asynchronously compiles a full scene description including:
    1. Obstacles detected (walls, poles, holes, people).
    2. Names of registered faces recognized in the current view.
    3. Running text recognition (OCR) on the central area of the frame.
    """
    audio.speak("Describing scene")
    
    # 1. Summarize obstacles
    obs_summary = ""
    if len(obstacles) > 0:
        counts = {}
        for obs in obstacles:
            lbl = obs["label"]
            pos = obs["position"]
            key_str = f"{lbl} {pos}" if pos != "ahead" else f"{lbl} ahead"
            counts[key_str] = counts.get(key_str, 0) + 1
            
        summary_items = []
        for key_str, count in counts.items():
            if count > 1:
                summary_items.append(f"{count} {key_str}s")
            else:
                summary_items.append(f"{key_str}")
        obs_summary = "I detect " + ", ".join(summary_items) + "."

    # 2. Summarize recognized faces
    face_summary = ""
    recognized_names = [f["name"] for f in detected_faces if f["name"] != "Unknown"]
    if len(recognized_names) > 0:
        face_summary = "I recognize " + " and ".join(set(recognized_names)) + "."

    # 3. Read text (OCR) in central region
    ocr_summary = ""
    if text_reader.reader:
        try:
            h, w = frame.shape[:2]
            crop_w = int(w * 0.6)
            crop_h = int(h * 0.6)
            x1 = (w - crop_w) // 2
            y1 = (h - crop_h) // 2
            roi = frame[y1:y1+crop_h, x1:x1+crop_w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            results = text_reader.reader.readtext(gray)
            text_blocks = [res[1] for res in results if res[2] > 0.3]
            detected_text = " ".join(text_blocks).strip()
            
            if detected_text:
                ocr_summary = f"There is text reading: '{detected_text}'."
                if logger:
                    logger.log_event("ocr", {"text": detected_text, "context": "scene_description"})
        except Exception as e:
            print(f"[SCENE DESCRIBE ERROR] OCR failed: {e}")

    # 4. Combine summaries
    parts = []
    if obs_summary:
        parts.append(obs_summary)
    if face_summary:
        parts.append(face_summary)
    if ocr_summary:
        parts.append(ocr_summary)
        
    if not parts:
        final_description = "Scene is clear. No obstacles, faces, or text detected."
    else:
        final_description = " ".join(parts)
        
    audio.speak(final_description, is_critical=True)
    print(f"[SCENE DESCRIPTION] {final_description}")

def main():
    print("===========================================")
    print("SightAssist: Advanced System Initialization")
    print("===========================================")

    # 1. Load Configurations
    config = load_config()
    if not config:
        print("Using default fallback settings.")
        config = {
            "cameras": {"input_source": 0, "resolution": {"width": 640, "height": 480}},
            "audio": {"speech_rate": 150, "volume": 1.0}
        }

    # 2. Initialize Audio Feedback
    audio_cfg = config.get("audio", {})
    audio = AudioFeedback(rate=audio_cfg.get("speech_rate", 125),
                          volume=audio_cfg.get("volume", 1.0),
                          cooldown=audio_cfg.get("cooldown", 4.0),
                          critical_cooldown=audio_cfg.get("critical_cooldown", 4.0),
                          close_cooldown=audio_cfg.get("close_cooldown", 8.0),
                          distance_update_threshold=audio_cfg.get("distance_update_threshold", 0.5),
                          global_silence_interval=audio_cfg.get("global_silence_interval", 2.5),
                          critical_silence_interval=audio_cfg.get("critical_silence_interval", 1.0))
    
    # 3. Initialize Logger First (so other modules can report to it)
    logger = ReportLogger()

    # 4. Initialize YOLO Vision Module
    ai_cfg = config.get("ai", {})
    obs_cfg = config.get("obstacles", {})
    try:
        vision_module = YoloVision(
            model_path=ai_cfg.get("yolo_model", "models/yolov8n.pt"),
            backend=ai_cfg.get("yolo_backend", "pytorch"),
            conf_thres=ai_cfg.get("confidence_threshold", 0.5),
            close_threshold=obs_cfg.get("close_threshold", 5.0),
            critical_threshold=obs_cfg.get("critical_threshold", 2.0),
            near_object_threshold=obs_cfg.get("near_object_threshold", 3.5),
            camera_height=obs_cfg.get("camera_height", 1.4),
            horizon_ratio=obs_cfg.get("horizon_ratio", 0.35),
            hole_sensitivity=obs_cfg.get("hole_sensitivity", "medium")
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize YOLO: {e}")
        vision_module = None

    # 5. Initialize Supplementary Modules
    color_id = ColorIdentifier()
    text_reader = TextReader()
    face_id = FaceRecognizer()
    emergency = EmergencyModule()

    # 6. Initialize Camera
    cam_cfg = config.get("cameras", {})
    camera = CameraModule(camera_index=cam_cfg.get("input_source", 0),
                          width=cam_cfg.get("resolution", {}).get("width", 640),
                          height=cam_cfg.get("resolution", {}).get("height", 480))
    
    if not camera.initialize():
        audio.speak_sync("Critical Error: Cannot access camera.")
        sys.exit(1)

    audio.speak_sync("SightAssist system initialized and fully operational.")
    
    # Track timers for visual confirmation overlays
    color_query_time = 0.0
    color_query_name = ""
    ocr_query_time = 0.0
    
    # Reading mode state
    reading_mode_active = False
    last_reading_ocr_time = 0.0
    
    # Keep track of recognized face greetings to prevent auditory spam
    last_face_greeting = {}  # name -> timestamp
    
    # Performance metrics
    frame_count = 0
    detected_faces = []
    
    # 7. Main Loop
    print("\n[MAIN] Entering real-time HUD loop. Focus window and press hotkeys:")
    print("  'Q' - Quit & Generate Report")
    print("  'C' - Identify Color in Reticle")
    print("  'T' - Read Text in Reticle")
    print("  'F' - Register Face")
    print("  'R' - Generate Report On-Demand")
    print("  'W' - Toggle Reading Mode (Continuous Text Reading)")
    print("  'E' - Describe Current Scene (Take Image Description)")
    print("  'Space' - Emergency Alert Toggle\n")

    try:
        while True:
            ret, frame = camera.get_frame()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            frame_count += 1
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            # Face recognition sub-loop (run face detection every 6 frames to sustain high FPS)
            if frame_count % 6 == 0:
                detected_faces = face_id.recognize_faces(frame)
                
                # Enforce cooldown on speaking face recognition warnings
                current_time = time.time()
                for face in detected_faces:
                    name = face["name"]
                    # If it's a recognized person
                    if name != "Unknown":
                        last_greet = last_face_greeting.get(name, 0.0)
                        if current_time - last_greet > 12.0:
                            audio.speak(f"{name} detected", is_critical=True)
                            last_face_greeting[name] = current_time
                            logger.log_event("face_recognition", {"name": name, "confidence": face["confidence"]})
                    else:
                        # Unknown person alert cooldown (15s)
                        last_unknown_greet = last_face_greeting.get("Unknown", 0.0)
                        if current_time - last_unknown_greet > 18.0:
                            audio.speak("Unknown person detected", is_critical=True)
                            last_face_greeting["Unknown"] = current_time
                            logger.log_event("face_recognition", {"name": "Unknown", "confidence": face["confidence"]})

            # AI processing steps
            if vision_module:
                annotated_frame, obstacles, infer_ms, fps, steer_advice = vision_module.process_frame(frame)
                
                # Check for automatic fall detection and camera abnormalities (black/stationary screen)
                emergency.check_fall_detection(obstacles, frame, audio=audio, logger=logger)
                emergency.check_camera_emergency(frame, obstacles, audio=audio, logger=logger)

                # Send steering instructions to audio feedback
                audio.speak_navigation(steer_advice)

                # Retrieve thresholds for warning log
                dist_cfg = config.get("obstacles", {})
                close_thres = dist_cfg.get("close_threshold", 5.0)
                critical_thres = dist_cfg.get("critical_threshold", 2.0)
                
                # Priority sorting for speech
                label_priority = {
                    "vehicle": 15, "car": 15, "truck": 15, "bus": 15, "motorcycle": 15, "tricycle": 15, "bicycle": 15,
                    "person": 12,
                    "pothole": 10, "pole": 8, "wall": 8, "cat": 5, "dog": 5
                }
                
                def get_priority(obs):
                    lbl_score = label_priority.get(obs["label"], 4)
                    pos_score = 2.0 if obs["position"] == "ahead" else 1.0
                    dist = obs.get("distance", 10.0)
                    dist_factor = 10.0 / max(0.5, dist)
                    return lbl_score * pos_score * dist_factor

                sorted_obstacles = sorted(obstacles, key=get_priority, reverse=True)
                
                # Speak the single highest-priority qualified obstacle that is not on cooldown
                audio.speak_obstacles(sorted_obstacles, close_threshold=close_thres, critical_threshold=critical_thres)
                
                # Log all close and critical obstacles to caregiver reports
                for obs in sorted_obstacles:
                    dist = obs["distance"]
                    if dist <= close_thres:
                        logger.log_event("obstacle_warning", {
                            "label": obs["label"],
                            "position": obs["position"],
                            "distance": dist,
                            "is_critical": (dist < critical_thres)
                        })
                        
                display_frame = annotated_frame
            else:
                display_frame = frame.copy()
                fps, infer_ms, steer_advice = 0.0, 0.0, "clear"
                obstacles = []

            # ─────────────────────────────────────────────────────────────────
            # PREMIUM HEADS-UP DISPLAY (HUD) LAYER
            # ─────────────────────────────────────────────────────────────────
            current_time = time.time()
            
            # If Reading Mode is active, run text recognition asynchronously every 1.5 seconds
            if reading_mode_active:
                if current_time - last_reading_ocr_time > 1.5:
                    text_reader.read_text_async(frame, audio, logger=logger, is_silent=True)
                    last_reading_ocr_time = current_time
            
            # 1. Top HUD Status Bar
            cv2.rectangle(display_frame, (0, 0), (w, 40), (12, 10, 18), -1)
            cv2.line(display_frame, (0, 40), (w, 40), (60, 60, 60), 1)
            
            # Status items
            status_text = "VISION: YOLOv8 | FACE ID: READY | OCR: READY | REPORT: LOGGING"
            if reading_mode_active:
                status_text = "VISION: YOLOv8 | FACE ID: READY | READING MODE: ACTIVE | REPORT: LOGGING"
            if emergency.active:
                status_text = "⚠️ EMERGENCY ASSISTANCE ACTIVE - CAREGIVER NOTIFIED"
                
            cv2.putText(display_frame, status_text, (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255) if not emergency.active else (0, 0, 255), 1, cv2.LINE_AA)
            
            # FPS Stats
            stats_text = f"FPS: {fps:.1f} | AI: {infer_ms:.1f}ms"
            cv2.putText(display_frame, stats_text, (w - 180, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # 2. Central Targeting Reticle
            reticle_color = (255, 255, 255)
            reticle_thickness = 1
            
            # Change color of reticle when a query happens to give visual confirmation
            if current_time - color_query_time < 2.0:
                reticle_color = (0, 255, 255)  # Yellow for color query
                reticle_thickness = 2
                # Display identified color name
                cv2.putText(display_frame, f"COLOR: {color_query_name.upper()}", 
                            (cx - 70, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            elif current_time - ocr_query_time < 2.0:
                reticle_color = (0, 255, 0)  # Green for OCR text query
                reticle_thickness = 2
                cv2.putText(display_frame, "OCR SCANNING...", 
                            (cx - 50, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw targeting crosshair circle (Radius 30)
            cv2.circle(display_frame, (cx, cy), 30, reticle_color, reticle_thickness)
            cv2.line(display_frame, (cx - 40, cy), (cx - 10, cy), reticle_color, reticle_thickness)
            cv2.line(display_frame, (cx + 10, cy), (cx + 40, cy), reticle_color, reticle_thickness)
            cv2.line(display_frame, (cx, cy - 40), (cx, cy - 10), reticle_color, reticle_thickness)
            cv2.line(display_frame, (cx, cy + 10), (cx, cy + 40), reticle_color, reticle_thickness)

            # 3. Draw Detected Faces Overlays
            for face in detected_faces:
                x, y, fw, fh = face["box"]
                name = face["name"]
                conf = face["confidence"]
                
                # Purple bounding box for Face recognition
                box_color = (182, 89, 155) if name != "Unknown" else (128, 128, 128)
                cv2.rectangle(display_frame, (x, y), (x + fw, y + fh), box_color, 2)
                
                # Draw name text
                label = f"{name}"
                if name != "Unknown":
                    label += f" ({conf:.2f})"
                cv2.putText(display_frame, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

            # 4. Path Directional Navigation Visual Indicators
            if steer_advice == "steer left":
                # Draw big arrow pointing left
                # Arrow points: (cx - 80, cy) -> (cx - 140, cy)
                cv2.arrowedLine(display_frame, (cx - 80, cy), (cx - 140, cy), (0, 255, 0), 5, tipLength=0.3)
                cv2.putText(display_frame, "STEER LEFT", (cx - 160, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            elif steer_advice == "steer right":
                # Draw big arrow pointing right
                cv2.arrowedLine(display_frame, (cx + 80, cy), (cx + 140, cy), (0, 255, 0), 5, tipLength=0.3)
                cv2.putText(display_frame, "STEER RIGHT", (cx + 90, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            elif steer_advice == "stop":
                # Draw a warning STOP badge in the center
                cv2.rectangle(display_frame, (cx - 60, cy - 80), (cx + 60, cy - 50), (0, 0, 255), -1)
                cv2.putText(display_frame, "STOP PATH BLOCKED", (cx - 55, cy - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            # 5. Flashing Emergency Distress Screen Border
            if emergency.active:
                border_flash = int(time.time() * 5) % 2
                border_color = (0, 0, 255) if border_flash else (0, 0, 100)
                cv2.rectangle(display_frame, (0, 0), (w, h), border_color, 10)
                cv2.putText(display_frame, "!!! EMERGENCY DISTRESS TRIGGERED !!!", (w // 2 - 190, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # Display window frame
            cv2.imshow('SightAssist Premium HUD', display_frame)

            # ─────────────────────────────────────────────────────────────────
            # KEYBOARD HOTKEY ROUTER
            # ─────────────────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[MAIN] Quit requested.")
                break
                
            elif key == ord('c'):
                # Color identification
                color_name, avg_bgr = color_id.identify_color(frame)
                color_query_name = color_name
                color_query_time = current_time
                audio.speak(f"Color {color_name}")
                logger.log_event("color_identification", {"color": color_name})
                
            elif key == ord('t'):
                # Text recognition (OCR) runs in background
                ocr_query_time = current_time
                text_reader.read_text_async(frame, audio, logger=logger)
                
            elif key == ord('f'):
                # Register face
                # Check if a face is currently detected in this frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_id.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
                
                if len(faces) == 0:
                    audio.speak("No face detected for registration. Please stand closer.")
                else:
                    # Select the largest detected face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    
                    audio.speak_sync("Registering face. Please enter details on screen.")
                    
                    # Spawn Tkinter simple name dialog
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    name = simpledialog.askstring("SightAssist Face ID", "Enter name of the registered person:")
                    root.destroy()
                    
                    if name:
                        name = name.strip()
                        success = face_id.register_face(frame, largest_face, name)
                        if success:
                            audio.speak(f"{name} registered successfully.")
                        else:
                            audio.speak("Face registration failed.")
                    else:
                        audio.speak("Face registration cancelled.")
                        
            elif key == ord(' '):
                # Emergency Toggle (Space only)
                if emergency.active:
                    emergency.clear_emergency(audio=audio)
                else:
                    emergency.trigger_emergency(frame, reason="Manual Alert", audio=audio, logger=logger)
                    
            elif key == ord('r'):
                # Generate caregiver report immediately
                audio.speak("Generating caregiver report.")
                md, html = logger.generate_report()
                if md:
                    audio.speak("Caregiver report generated.")
                else:
                    audio.speak("Report generation failed.")
                    
            elif key == ord('w'):
                # Toggle continuous reading mode
                reading_mode_active = not reading_mode_active
                if reading_mode_active:
                    audio.speak("Reading mode active", is_critical=True)
                else:
                    audio.speak("Reading mode inactive", is_critical=True)
                    
            elif key == ord('e'):
                # Describe what the system sees (take an image description)
                t = threading.Thread(
                    target=describe_scene_async,
                    args=(frame.copy(), obstacles, detected_faces, audio, text_reader, logger)
                )
                t.daemon = True
                t.start()

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")
    
    finally:
        # Cleanup and write report
        print("[MAIN] Cleaning up resources and generating caregiver report...")
        camera.release()
        cv2.destroyAllWindows()
        
        # Speak final shutdown sequence
        audio.speak_sync("SightAssist system shutting down. compiling final caregiver reports.")
        
        # Release audio resources (stops thread, clears queue)
        audio.release()
        
        # Compile reports
        md_path, html_path = logger.generate_report()
        
        print(f"[MAIN] Caregiver reports saved successfully.\nShutdown complete.")

if __name__ == "__main__":
    main()
