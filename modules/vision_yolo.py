import os
import time
import numpy as np
import cv2

class YoloVision:
    def __init__(self, model_path, backend="pytorch", conf_thres=0.5,
                 close_threshold=5.0, critical_threshold=2.0, near_object_threshold=3.5,
                 camera_height=1.4, horizon_ratio=0.35, hole_sensitivity="medium"):
        from ultralytics import YOLO
        
        self.model_path = model_path
        self.backend = backend.lower()
        self.conf_thres = conf_thres
        
        self.close_threshold = close_threshold
        self.critical_threshold = critical_threshold
        self.near_object_threshold = near_object_threshold
        self.H_cam = camera_height
        self.y0_ratio = horizon_ratio
        self.hole_sensitivity = hole_sensitivity.lower()
        
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

    def estimate_distance(self, label, box, frame_shape):
        """Estimate the distance of an object using bounding box height and ground projection fallbacks."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        
        # Focal length estimate (assume ~60 degree horizontal FOV, F approx width)
        F = w
        
        # 1. Ground plane estimation (calibrated linear-inverse mapping)
        # Maps y2 = h to 1.0 meter (very close), and y2 = y0 to 10.0+ meters (far away).
        # This handles arbitrary camera tilts much better than a rigid horizontal assumption.
        y0 = int(self.y0_ratio * h)
        if y2 <= y0:
            d_ground = 100.0
        else:
            y_norm = (y2 - y0) / (h - y0)
            y_norm = np.clip(y_norm, 0.01, 1.0)
            d_ground = 1.0 + 2.0 * (1.0 - y_norm) / y_norm
        
        # 2. Bounding box height size-based estimation
        real_heights = {
            "person": 1.7,
            "vehicle": 1.6,
            "tricycle": 1.5,
            "bicycle": 1.0,
            "pole": 2.5,
            "hole": 0.5,
            
            "chair": 0.8,
            "couch": 0.8,
            "bed": 0.6,
            "dining table": 0.75,
            "toilet": 0.75,
            "bench": 0.8,
            
            "dog": 0.5,
            "cat": 0.3,
            "horse": 1.5,
            "sheep": 0.8,
            "cow": 1.4,
            "elephant": 2.5,
            "bear": 1.5,
            "zebra": 1.4,
            "giraffe": 4.0,
            
            "fire hydrant": 0.8,
            "stop sign": 1.0,
            "parking meter": 1.2,
            "traffic light": 1.5,
            
            "suitcase": 0.7,
            "backpack": 0.5,
            "handbag": 0.3,
            "umbrella": 1.0,
        }
        
        real_h = real_heights.get(label, 1.0)
        box_h = max(1, y2 - y1)
        d_size = (real_h * F) / box_h
        
        if label in {"hole", "pothole"}:
            distance = d_ground
        else:
            # We take the minimum of both to be safe and conservative.
            # E.g. if the object is cut off at the top (head cropped), the bottom ground projection remains accurate.
            # If the object's bottom is cut off, the size-based estimation helps.
            distance = min(d_ground, d_size)
            
        return max(0.1, min(100.0, float(distance)))

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
        
        # Optimization: Downscale frame for heuristics to reduce CPU load (4x to 16x speedup)
        scale = 320.0 / w
        sh, sw = int(h * scale), 320
        small_frame = cv2.resize(frame, (sw, sh))
        
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Dilate vertical structures to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poles = []
        
        # Scale dimensions and coordinates
        min_ch = 60.0 * scale
        max_ch = sh * 0.7
        
        scaled_yolo_boxes = []
        for yb in yolo_boxes:
            scaled_yolo_boxes.append([
                int(yb[0] * scale),
                int(yb[1] * scale),
                int(yb[2] * scale),
                int(yb[3] * scale)
            ])
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            # Poles are vertical, thin and tall
            if ch > min_ch and 2 <= cw < (40 * scale) and 4.5 < (ch / cw) < 12.0:
                if y > sh * 0.2 and ch < max_ch:
                    if y + ch > sh * 0.35 and y < sh * 0.85:
                        box = [x, y, x + cw, y + ch]
                        if not self.is_overlapping(box, scaled_yolo_boxes):
                            box_orig = [
                                int(x / scale),
                                int(y / scale),
                                int((x + cw) / scale),
                                int((y + ch) / scale)
                            ]
                            poles.append(box_orig)
                            if len(poles) >= 3:
                                break
        return poles

    def detect_holes(self, frame, yolo_boxes):
        """Heuristic dark contour detector in the road region to locate potholes or ground openings."""
        h, w = frame.shape[:2]
        
        # Optimization: Downscale frame
        scale = 320.0 / w
        sh, sw = int(h * scale), 320
        small_frame = cv2.resize(frame, (sw, sh))
        
        # Restrict search area to the lower 45% (road plane)
        road_y_start = int(sh * 0.55)
        road_roi = small_frame[road_y_start:sh, 0:sw]
        
        if road_roi.size == 0:
            return []
            
        gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Calculate standard deviation and mean road intensity
        road_mean = np.mean(blur)
        road_std = np.std(blur)
        
        if self.hole_sensitivity == "high":
            base_offset = 35
            std_mult = 1.0
        elif self.hole_sensitivity == "low":
            base_offset = 65
            std_mult = 2.0
        elif self.hole_sensitivity == "very_low":
            base_offset = 80
            std_mult = 2.5
        else:  # "medium" or default
            base_offset = 50
            std_mult = 1.5
            
        offset = max(base_offset, int(road_std * std_mult))
        thresh_val = max(10, int(road_mean - offset))
        _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Merge fragment contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        holes = []
        
        scaled_yolo_boxes = []
        for yb in yolo_boxes:
            scaled_yolo_boxes.append([
                int(yb[0] * scale),
                int(yb[1] * scale),
                int(yb[2] * scale),
                int(yb[3] * scale)
            ])
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            
            if x < sw * 0.1 or x + cw > sw * 0.9:
                continue
                
            y2_full = y + ch + road_y_start
            y_norm = (y2_full - road_y_start) / (sh - road_y_start) if sh > road_y_start else 0.0
            y_norm = np.clip(y_norm, 0.0, 1.0)
            
            # Scale min area
            scale_sq = scale ** 2
            dynamic_min_area = (400 + (y_norm ** 2) * 1600) * scale_sq
            dynamic_max_area = 8000 * scale_sq
            
            # Verify minimum dimensions on small frame
            if dynamic_min_area < area < dynamic_max_area and cw >= 8 and ch >= 6:
                aspect_ratio = cw / ch if ch > 0 else 0
                # Strict Aspect Ratio Check (Potholes are round/oval projection, no long lines/borders)
                if 1.0 <= aspect_ratio <= 2.2:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # Strict Solidity Check (Actual potholes are highly defined and convex)
                    if solidity > 0.92:
                        # Contrast Check: Pothole center must be significantly darker than surrounding road
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        mean_contour_val = cv2.mean(gray, mask=mask)[0]
                        
                        if mean_contour_val < road_mean * 0.70:
                            box_small = [x, y + road_y_start, x + cw, y + ch + road_y_start]
                            if not self.is_overlapping(box_small, scaled_yolo_boxes):
                                box_orig = [
                                    int(box_small[0] / scale),
                                    int(box_small[1] / scale),
                                    int(box_small[2] / scale),
                                    int(box_small[3] / scale)
                                ]
                                holes.append(box_orig)
                                if len(holes) >= 3:
                                    break
        return holes

    def detect_walls(self, frame, yolo_boxes):
        """Heuristic wall detector utilizing line segments on left/right frame margins."""
        h, w = frame.shape[:2]
        
        # Optimization: Downscale frame
        scale = 320.0 / w
        sh, sw = int(h * scale), 320
        small_frame = cv2.resize(frame, (sw, sh))
        
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        
        min_line_len = int(80 * scale)
        max_line_gap = int(15 * scale)
        
        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=min_line_len, maxLineGap=max_line_gap)
        walls = []
        
        scaled_yolo_boxes = []
        for yb in yolo_boxes:
            scaled_yolo_boxes.append([
                int(yb[0] * scale),
                int(yb[1] * scale),
                int(yb[2] * scale),
                int(yb[3] * scale)
            ])
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                
                # Near-vertical or steep diagonal lines
                if abs(dx) < 5 or abs(dy / dx) > 1.2:
                    mx = (x1 + x2) / 2.0
                    # Limit to left 25% or right 25% of the frame
                    if mx < sw * 0.25 or mx > sw * 0.75:
                        bx1, bx2 = min(x1, x2), max(x1, x2)
                        by1, by2 = min(y1, y2), max(y1, y2)
                        
                        if (bx2 - bx1) < 15 * scale:
                            bx1 = max(0, bx1 - int(10 * scale))
                            bx2 = min(sw, bx2 + int(10 * scale))
                            
                        box_small = [bx1, by1, bx2, by2]
                        if not self.is_overlapping(box_small, scaled_yolo_boxes):
                            box_orig = [
                                int(bx1 / scale),
                                int(by1 / scale),
                                int(bx2 / scale),
                                int(by2 / scale)
                            ]
                            walls.append(box_orig)
                            if len(walls) >= 2:
                                break
        return walls


    def process_frame(self, frame):
        """Runs the chosen YOLO model + CV heuristics on a frame, returning the annotated frame, obstacles, and path steering advice."""
        start_time = time.time()
        h, w = frame.shape[:2]
        
        # 1. Run YOLO inference
        results = self.model(frame, conf=self.conf_thres, verbose=False)
        
        elapsed_ms = (time.time() - start_time) * 1000.0
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

        detected_obstacles = []
        yolo_boxes = []
        
        vehicle_classes = {"car", "truck", "bus"}
        
        # Target general objects to report when they are near (D < self.near_object_threshold)
        general_obstacle_classes = {
            # Animals
            "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            # Furniture / large items
            "chair", "couch", "bed", "dining table", "toilet", "bench", "potted plant",
            # Outdoor features
            "fire hydrant", "stop sign", "parking meter", "traffic light",
            # Luggage / bags
            "suitcase", "backpack", "handbag", "umbrella"
        }

        # 2. Extract and map YOLO boxes
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            raw_name = self.model.names[class_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            
            yolo_boxes.append([x1, y1, x2, y2])
            
            mapped_label = None
            is_primary = False
            
            if raw_name == "person":
                mapped_label = "person"
                is_primary = True
            elif raw_name in vehicle_classes:
                mapped_label = "vehicle"
                is_primary = True
            elif raw_name == "motorcycle":
                wb = x2 - x1
                hb = y2 - y1
                mapped_label = "tricycle" if (hb > 0 and (wb / hb) > 0.85) else "vehicle"
                is_primary = True
            elif raw_name == "bicycle":
                wb = x2 - x1
                hb = y2 - y1
                mapped_label = "tricycle" if (hb > 0 and (wb / hb) > 0.85) else "bicycle"
                is_primary = True
            elif raw_name == "cat":
                mapped_label = "cat"
                is_primary = True
            elif raw_name in general_obstacle_classes:
                mapped_label = raw_name
                is_primary = False
                
            if mapped_label:
                # Estimate distance
                dist = self.estimate_distance(mapped_label, [x1, y1, x2, y2], frame.shape)
                
                # Keep primary classes always, and general objects only if they are near
                if is_primary or (dist <= self.near_object_threshold):
                    pos = self.get_spatial_position([x1, y1, x2, y2], w)
                    detected_obstacles.append({
                        "label": mapped_label,
                        "position": pos,
                        "box": [x1, y1, x2, y2],
                        "confidence": conf,
                        "distance": dist
                    })

        # 3. Detect custom obstacles (Poles, Holes, Walls)
        poles = self.detect_poles(frame, yolo_boxes)
        for box in poles:
            pos = self.get_spatial_position(box, w)
            dist = self.estimate_distance("pole", box, frame.shape)
            detected_obstacles.append({
                "label": "pole",
                "position": pos,
                "box": box,
                "confidence": 0.70,
                "distance": dist
            })

        holes = self.detect_holes(frame, yolo_boxes)
        for box in holes:
            pos = self.get_spatial_position(box, w)
            dist = self.estimate_distance("pothole", box, frame.shape)
            detected_obstacles.append({
                "label": "pothole",
                "position": pos,
                "box": box,
                "confidence": 0.75,
                "distance": dist
            })

        walls = self.detect_walls(frame, yolo_boxes)
        for box in walls:
            pos = self.get_spatial_position(box, w)
            dist = self.estimate_distance("wall", box, frame.shape)
            detected_obstacles.append({
                "label": "wall",
                "position": pos,
                "box": box,
                "confidence": 0.70,
                "distance": dist
            })

        # 4. Path analysis & Navigation advice (Steering heuristics)
        # Determine if there is a critical threat ahead
        blocked_ahead = False
        left_dist = 100.0
        right_dist = 100.0
        
        for obs in detected_obstacles:
            pos = obs["position"]
            dist = obs["distance"]
            if pos == "ahead" and dist < self.critical_threshold:
                blocked_ahead = True
            elif pos == "on the left" and dist < left_dist:
                left_dist = dist
            elif pos == "on the right" and dist < right_dist:
                right_dist = dist
                
        steer_advice = "clear"
        if blocked_ahead:
            # Check which side is clearer
            if left_dist >= 3.5 and right_dist >= 3.5:
                # Both clear, steer left by default
                steer_advice = "steer left"
            elif left_dist > right_dist:
                steer_advice = "steer left"
            elif right_dist > left_dist:
                steer_advice = "steer right"
            else:
                # Both blocked
                steer_advice = "stop"

        # 5. Premium Drawing Overlays
        annotated_frame = frame.copy()
        color_map = {
            "person": (113, 204, 46),    # Emerald Green (BGR)
            "vehicle": (219, 152, 52),   # Blue (BGR)
            "tricycle": (182, 89, 155),  # Purple (BGR)
            "bicycle": (156, 188, 26),   # Turquoise (BGR)
            "pole": (15, 196, 241),      # Yellow (BGR)
            "hole": (60, 76, 231),       # Red (BGR)
            "wall": (240, 16, 240),      # Magenta (BGR)
            "cat": (74, 195, 236)        # Warm Amber (BGR)
        }

        for obs in detected_obstacles:
            label = obs["label"]
            pos = obs["position"]
            x1, y1, x2, y2 = obs["box"]
            dist = obs.get("distance", 99.0)
            
            is_critical = dist < self.critical_threshold
            if is_critical:
                color = (0, 0, 255)  # Vibrant Warning Red
                thickness = 3
            else:
                color = color_map.get(label, (0, 165, 255))
                thickness = 2
            
            # Draw standard bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Corner markers for premium HUD aesthetic
            length = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            if length > 0:
                cv2.line(annotated_frame, (x1, y1), (x1 + length, y1), color, thickness + 2)
                cv2.line(annotated_frame, (x1, y1), (x1, y1 + length), color, thickness + 2)
                cv2.line(annotated_frame, (x2, y1), (x2 - length, y1), color, thickness + 2)
                cv2.line(annotated_frame, (x2, y1), (x2, y1 + length), color, thickness + 2)
                cv2.line(annotated_frame, (x1, y2), (x1 + length, y2), color, thickness + 2)
                cv2.line(annotated_frame, (x1, y2), (x1, y2 - length), color, thickness + 2)
                cv2.line(annotated_frame, (x2, y2), (x2 - length, y2), color, thickness + 2)
                cv2.line(annotated_frame, (x2, y2), (x2, y2 - length), color, thickness + 2)
            
            # Overlay Text Label with distance
            dist_str = f" | {dist:.1f}m" if dist < 100.0 else ""
            warning_prefix = "CRITICAL: " if is_critical else ""
            text = f"{warning_prefix}{label.upper()} ({pos}){dist_str}"
            
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated_frame, text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated_frame, detected_obstacles, elapsed_ms, fps, steer_advice

