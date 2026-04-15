import os
import time

class YoloVision:
    def __init__(self, model_path, backend="pytorch", conf_thres=0.5):
        from ultralytics import YOLO
        
        self.model_path = model_path
        self.backend = backend.lower()
        self.conf_thres = conf_thres
        
        # If ONNX is requested, adjust the path to point to .onnx if it exists
        if self.backend == "onnx":
            base, ext = os.path.splitext(self.model_path)
            onnx_path = base + ".onnx"
            if os.path.exists(onnx_path):
                self.model_path = onnx_path
                print(f"[YOLO] Using ONNX Lite model: {self.model_path}")
            else:
                print(f"[YOLO] WARNING: ONNX model {onnx_path} not found. Ensure it was exported. Falling back to {self.model_path}")

        print(f"[YOLO] Loading model {self.model_path}...")
        self.model = YOLO(self.model_path, task='detect')
        print("[YOLO] Model loaded successfully.")

    def process_frame(self, frame):
        """Runs the chosen YOLO model on a frame, returning the processed frame (with drawing) and info."""
        start_time = time.time()
        
        # Run inference
        results = self.model(frame, conf=self.conf_thres, verbose=False)
        
        elapsed_ms = (time.time() - start_time) * 1000.0
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

        # Draw results on the frame
        annotated_frame = results[0].plot()
        
        # Detect obstacles (very rudimentarily looking at object names)
        detected_objects = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            name = self.model.names[class_id]
            detected_objects.append(name)
            
        return annotated_frame, detected_objects, elapsed_ms, fps
