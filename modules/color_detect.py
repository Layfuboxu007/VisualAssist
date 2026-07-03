import cv2
import numpy as np

class ColorIdentifier:
    def __init__(self):
        pass

    def identify_color(self, frame, box_size=60):
        """
        Samples the central box_size x box_size pixel area of the frame,
        determines the average/dominant color, and returns:
          - color_name: a string (e.g. "bright red", "dark green", "gray")
          - rgb_value: BGR tuple of the average color (for drawing/rendering)
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Define region of interest (ROI) centered in the frame
        x1 = max(0, cx - box_size // 2)
        y1 = max(0, cy - box_size // 2)
        x2 = min(w, cx + box_size // 2)
        y2 = min(h, cy + box_size // 2)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown", (128, 128, 128)
            
        # Calculate the average BGR color of the ROI
        avg_bgr = cv2.mean(roi)[:3]
        avg_bgr_int = (int(avg_bgr[0]), int(avg_bgr[1]), int(avg_bgr[2]))
        
        # Convert ROI to HSV to analyze color attributes
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Circular averaging of Hue to prevent wrap-around errors at 0/180
        h_channel = hsv_roi[:, :, 0].astype(np.float32)
        s_channel = hsv_roi[:, :, 1].astype(np.float32)
        v_channel = hsv_roi[:, :, 2].astype(np.float32)
        
        # Convert hue to radians for circular mean: Hue goes from 0-179 -> map to 0-2pi
        hue_rad = (h_channel / 180.0) * 2.0 * np.pi
        mean_cos = np.mean(np.cos(hue_rad))
        mean_sin = np.mean(np.sin(hue_rad))
        
        mean_hue_rad = np.arctan2(mean_sin, mean_cos)
        if mean_hue_rad < 0:
            mean_hue_rad += 2.0 * np.pi
        avg_h = (mean_hue_rad / (2.0 * np.pi)) * 180.0
        
        avg_s = np.mean(s_channel)
        avg_v = np.mean(v_channel)
        
        # Determine descriptive color name
        color_name = self._classify_hsv(avg_h, avg_s, avg_v)
        return color_name, avg_bgr_int

    def _classify_hsv(self, h, s, v):
        """Helper to classify HSV values into a human-friendly string."""
        # 1. Neutral colors (low saturation)
        if s < 30:
            if v > 200:
                return "white"
            elif v < 50:
                return "black"
            else:
                return "gray"
                
        # 2. Very low brightness is always black
        if v < 40:
            return "black"
            
        # 3. Saturation and value modifiers
        prefix = ""
        if v > 220 and s > 170:
            prefix = "bright "
        elif v < 110:
            prefix = "dark "
        elif s < 90:
            prefix = "pale "
            
        # 4. Classify Hue
        # Hue ranges in OpenCV are 0-179.
        if h < 8 or h >= 168:
            # Red/Brown distinction based on brightness and saturation
            if v < 120 and s < 130:
                return "brown"
            return prefix + "red"
        elif h < 20:
            if v < 120 and s < 130:
                return "brown"
            return prefix + "orange"
        elif h < 35:
            if v < 90 and s < 110:
                return "brown"
            return prefix + "yellow"
        elif h < 82:
            return prefix + "green"
        elif h < 128:
            return prefix + "blue"
        elif h < 145:
            return prefix + "purple"
        else: # 145 <= h < 168
            return prefix + "pink"
