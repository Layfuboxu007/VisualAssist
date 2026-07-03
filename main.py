import cv2
import yaml
import sys
import time

from modules.camera import CameraModule
from modules.audio_feedback import AudioFeedback
from modules.vision_yolo import YoloVision

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Could not load config: {e}")
        return None

def main():
    print("===========================================")
    print("SightAssist: Beta Framework Initialization")
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
    
    # 3. Initialize YOLO Vision Module
    ai_cfg = config.get("ai", {})
    obs_cfg = config.get("obstacles", {})
    try:
        vision_module = YoloVision(
            model_path=ai_cfg.get("yolo_model", "yolov8n.pt"),
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
    
    # We use a synchronous speak here just so it's fully heard before the loop starts
    audio.speak_sync("SightAssist framework initializing.")

    # 3. Initialize Camera
    cam_cfg = config.get("cameras", {})
    camera = CameraModule(camera_index=cam_cfg.get("input_source", 0),
                          width=cam_cfg.get("resolution", {}).get("width", 640),
                          height=cam_cfg.get("resolution", {}).get("height", 480))
    
    if not camera.initialize():
        audio.speak_sync("Critical Error: Cannot access camera.")
        sys.exit(1)

    audio.speak_sync("System ready. Press 'Q' to quit.")

    # 4. Main Processing Loop
    print("\n[MAIN] Entering main processing loop. Close the window or press 'q' to exit.")
    
    try:
        while True:
            ret, frame = camera.get_frame()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            # AI processing steps
            if vision_module:
                annotated_frame, obstacles, infer_ms, fps = vision_module.process_frame(frame)
                
                # Define obstacle warning priority (higher = more critical)
                label_priority = {
                    "hole": 10,
                    "person": 9,
                    "vehicle": 8,
                    "tricycle": 7,
                    "bicycle": 6,
                    "pole": 5,
                    "cat": 5
                }
                
                # Retrieve threshold constants from config for verbal alerts
                dist_cfg = config.get("obstacles", {})
                close_thres = dist_cfg.get("close_threshold", 5.0)
                critical_thres = dist_cfg.get("critical_threshold", 2.0)
                
                def get_priority(obs):
                    lbl_score = label_priority.get(obs["label"], 4)  # Default to 4 for general objects
                    pos_score = 2.0 if obs["position"] == "ahead" else 1.0
                    dist = obs.get("distance", 10.0)
                    dist_factor = 10.0 / max(0.5, dist)
                    return lbl_score * pos_score * dist_factor

                # Sort obstacles so the most critical and closest items are spoken first
                sorted_obstacles = sorted(obstacles, key=get_priority, reverse=True)
                
                for obs in sorted_obstacles:
                    audio.speak_obstacle(
                        label=obs["label"],
                        position=obs["position"],
                        distance=obs.get("distance", 99.0),
                        close_threshold=close_thres,
                        critical_threshold=critical_thres
                    )
                    
                display_frame = annotated_frame
                # Draw FPS Performance (removed PyTorch backend string)
                cv2.putText(display_frame, f"FPS: {fps:.1f} | ms: {infer_ms:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                display_frame = frame.copy()
                cv2.putText(display_frame, "SightAssist Beta", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(display_frame, "Press 'Q' to Quit", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display the resulting frame
            cv2.imshow('SightAssist Beta', display_frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[MAIN] Quit signal received.")
                break
                
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")
    
    finally:
        # 5. Cleanup
        print("[MAIN] Cleaning up resources...")
        camera.release()
        cv2.destroyAllWindows()
        audio.speak_sync("SightAssist system shutting down.")
        print("[MAIN] Shutdown complete.")

if __name__ == "__main__":
    main()
