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
    print("SmartVision: Beta Framework Initialization")
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
    audio = AudioFeedback(rate=audio_cfg.get("speech_rate", 150),
                          volume=audio_cfg.get("volume", 1.0))
    
    # 3. Initialize YOLO Vision Module
    ai_cfg = config.get("ai", {})
    try:
        vision_module = YoloVision(
            model_path=ai_cfg.get("yolo_model", "yolov8n.pt"),
            backend=ai_cfg.get("yolo_backend", "pytorch"),
            conf_thres=ai_cfg.get("confidence_threshold", 0.5)
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize YOLO: {e}")
        vision_module = None

    
    # We use a synchronous speak here just so it's fully heard before the loop starts
    audio.speak_sync("Smart Vision framework initializing.")

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
                annotated_frame, objects, infer_ms, fps = vision_module.process_frame(frame)
                
                # Check for "person" or critical obstacles to speak
                if "person" in objects:  # simple detection trigger for testing
                    audio.speak("Person ahead.")
                    
                display_frame = annotated_frame
                # Draw FPS Performance
                cv2.putText(display_frame, f"Backend: {vision_module.backend.upper()} | FPS: {fps:.1f} | ms: {infer_ms:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                display_frame = frame.copy()
                cv2.putText(display_frame, "SmartVision Beta", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(display_frame, "Press 'Q' to Quit", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display the resulting frame
            cv2.imshow('SmartVision Beta', display_frame)

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
        audio.speak_sync("Smart vision system shutting down.")
        print("[MAIN] Shutdown complete.")

if __name__ == "__main__":
    main()
