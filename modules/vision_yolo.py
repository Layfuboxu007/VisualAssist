import os
import time
import numpy as np
import cv2

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

    def get_spatial_position(self, box, width):
        """Determine if an object is on the left, ahead, or on the right."""
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2.0
        if center_x < width * 0.35:
            return "on the left"
        elif center_x > width * 0.65:
            return "on the right"
        else:
            return "ahead"

    def is_overlapping(self, box, yolo_boxes):
        """Check if the center of a box is inside any of the YOLO bounding boxes."""
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        for ybox in yolo_boxes:
            if ybox[0] <= cx <= ybox[2] and ybox[1] <= cy <= ybox[3]:
                return True
        return False

    def detect_poles(self, frame, yolo_boxes):
        """Heuristic vertical edge/contour detector to locate vertical poles and posts."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Dilate vertical structures to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poles = []
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            # Poles are vertical: high height-to-width ratio, relatively thin and tall
            if ch > 60 and cw < 45 and (ch / cw) > 4.5:
                # Spanning from middle to bottom region
                if y + ch > h * 0.3 and y < h * 0.9:
                    box = [x, y, x + cw, y + ch]
                    if not self.is_overlapping(box, yolo_boxes):
                        poles.append(box)
                        if len(poles) >= 3:  # Limit detections to avoid HUD clutter
                            break
        return poles

    def detect_holes(self, frame, yolo_boxes):
        """Heuristic dark contour detector in the road region to locate potholes or ground openings."""
        h, w = frame.shape[:2]
        # Restrict search area to the lower 45% (road plane)
        road_y_start = int(h * 0.55)
        road_roi = frame[road_y_start:h, 0:w]
        
        if road_roi.size == 0:
            return []
            
        gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Calculate dark threshold dynamically relative to the local average road intensity
        road_mean = np.mean(blur)
        thresh_val = max(10, int(road_mean - 35))
        _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Merge fragment contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        holes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 150 < area < 8000:
                x, y, cw, ch = cv2.boundingRect(contour)
                # Holes on the ground map as flat horizontal ellipses (width > height) due to perspective
                if cw / ch > 1.2:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    if solidity > 0.6:
                        # Translate ROI coordinates back to full frame
                        box = [x, y + road_y_start, x + cw, y + ch + road_y_start]
                        if not self.is_overlapping(box, yolo_boxes):
                            holes.append(box)
                            if len(holes) >= 3:
                                break
        return holes

    def process_frame(self, frame):
        """Runs the chosen YOLO model + CV heuristics on a frame, returning the annotated frame and list of obstacles."""
        start_time = time.time()
        h, w = frame.shape[:2]
        
        # 1. Run YOLO inference
        results = self.model(frame, conf=self.conf_thres, verbose=False)
        
        elapsed_ms = (time.time() - start_time) * 1000.0
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

        detected_obstacles = []
        yolo_boxes = []
        
        vehicle_classes = {"car", "truck", "bus"}

        # 2. Extract and map YOLO boxes
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            raw_name = self.model.names[class_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            
            yolo_boxes.append([x1, y1, x2, y2])
            
            mapped_label = None
            if raw_name == "person":
                mapped_label = "person"
            elif raw_name in vehicle_classes:
                mapped_label = "vehicle"
            elif raw_name == "motorcycle":
                # Tricycle heuristic: if it has a wide aspect ratio (width / height > 0.85)
                wb = x2 - x1
                hb = y2 - y1
                mapped_label = "tricycle" if (hb > 0 and (wb / hb) > 0.85) else "vehicle"
            elif raw_name == "bicycle":
                wb = x2 - x1
                hb = y2 - y1
                mapped_label = "tricycle" if (hb > 0 and (wb / hb) > 0.85) else "bicycle"
                
            if mapped_label:
                pos = self.get_spatial_position([x1, y1, x2, y2], w)
                detected_obstacles.append({
                    "label": mapped_label,
                    "position": pos,
                    "box": [x1, y1, x2, y2],
                    "confidence": conf
                })

        # 3. Detect custom obstacles (Poles and Holes)
        poles = self.detect_poles(frame, yolo_boxes)
        for box in poles:
            pos = self.get_spatial_position(box, w)
            detected_obstacles.append({
                "label": "pole",
                "position": pos,
                "box": box,
                "confidence": 0.70
            })

        holes = self.detect_holes(frame, yolo_boxes)
        for box in holes:
            pos = self.get_spatial_position(box, w)
            detected_obstacles.append({
                "label": "hole",
                "position": pos,
                "box": box,
                "confidence": 0.75
            })

        # 4. Premium Drawing Overlays
        annotated_frame = frame.copy()
        color_map = {
            "person": (113, 204, 46),    # Emerald Green (BGR)
            "vehicle": (219, 152, 52),   # Blue (BGR)
            "tricycle": (182, 89, 155),  # Purple (BGR)
            "bicycle": (156, 188, 26),   # Turquoise (BGR)
            "pole": (15, 196, 241),      # Yellow (BGR)
            "hole": (60, 76, 231)        # Red (BGR)
        }

        for obs in detected_obstacles:
            label = obs["label"]
            pos = obs["position"]
            x1, y1, x2, y2 = obs["box"]
            color = color_map.get(label, (0, 255, 0))
            
            # Standard rectangle outline
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # High-end corner markers
            length = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            if length > 0:
                cv2.line(annotated_frame, (x1, y1), (x1 + length, y1), color, 4)
                cv2.line(annotated_frame, (x1, y1), (x1, y1 + length), color, 4)
                cv2.line(annotated_frame, (x2, y1), (x2 - length, y1), color, 4)
                cv2.line(annotated_frame, (x2, y1), (x2, y1 + length), color, 4)
                cv2.line(annotated_frame, (x1, y2), (x1 + length, y2), color, 4)
                cv2.line(annotated_frame, (x1, y2), (x1, y2 - length), color, 4)
                cv2.line(annotated_frame, (x2, y2), (x2 - length, y2), color, 4)
                cv2.line(annotated_frame, (x2, y2), (x2, y2 - length), color, 4)
            
            # Overlay Text Label
            text = f"{label.upper()} ({pos})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated_frame, text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated_frame, detected_obstacles, elapsed_ms, fps

