"""
vision_tflite.py
-----------------
TensorFlow Lite object detection module for SmartVision.
Uses a standard SSD / EfficientDet-Lite .tflite model with COCO-style labels.
Compatible with both Windows (development) and Raspberry Pi (deployment).
"""

import os
import time
import numpy as np
import cv2


# ── TFLite runtime import (platform-aware) ───────────────────────────────────
def _load_tflite_interpreter():
    """Return the tflite Interpreter class from whichever package is available.

    Priority order:
      1. ai_edge_litert  — Google's new standalone package (no deprecation warning)
      2. tflite_runtime  — Lightweight runtime for Raspberry Pi / Linux
      3. tensorflow      — Full TF wheel (Windows/macOS dev machines)
    """
    try:
        # New Google package (TF 2.20+ replacement for tflite-runtime)
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        # Lightweight runtime (Raspberry Pi / Linux)
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        import warnings
        # Suppress the deprecation warning from TF >= 2.20
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            from tensorflow.lite.python.interpreter import Interpreter
        return Interpreter
    except ImportError:
        raise ImportError(
            "No TFLite runtime found. Install one of:\n"
            "  pip install ai-edge-litert          (Raspberry Pi / Linux / Windows)\n"
            "  pip install tflite-runtime          (older Raspberry Pi / Linux)\n"
            "  pip install tensorflow              (Windows / macOS dev)"
        )


class TFLiteVision:
    """
    Runs TFLite object detection inference on camera frames.

    Expected model: SSD MobileNet V2 / EfficientDet-Lite (COCO, 80 classes).
    Both can be downloaded from the TensorFlow Model Zoo or converted via
    `tflite_model_maker`.

    Args:
        model_path  (str):  Path to the .tflite model file.
        labels_path (str):  Path to the COCO labels text file (one label per line).
        conf_thres  (float): Minimum confidence score to keep a detection.
        num_threads (int):   Number of CPU threads for inference (default 4).
    """

    def __init__(self, model_path: str, labels_path: str,
                 conf_thres: float = 0.5, num_threads: int = 4):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[TFLite] Model not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"[TFLite] Labels not found: {labels_path}")

        self.conf_thres = conf_thres

        # ── Load labels ──────────────────────────────────────────────────────
        with open(labels_path, "r") as f:
            # Strip index prefix if present (e.g. "1  person")
            self.labels = [
                line.strip().split(maxsplit=1)[-1]
                for line in f.readlines()
                if line.strip()
            ]
        print(f"[TFLite] Loaded {len(self.labels)} labels from {labels_path}")

        # ── Load TFLite model ─────────────────────────────────────────────────
        import warnings
        Interpreter = _load_tflite_interpreter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.interpreter = Interpreter(
                model_path=model_path,
                num_threads=num_threads
            )
        self.interpreter.allocate_tensors()

        # ── Inspect input / output tensors ────────────────────────────────────
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        inp = self.input_details[0]
        self.input_height = inp["shape"][1]
        self.input_width  = inp["shape"][2]
        self.is_floating_model = (inp["dtype"] == np.float32)

        print(f"[TFLite] Model loaded: {model_path}")
        print(f"[TFLite] Input size : {self.input_width}×{self.input_height}  "
              f"({'float32' if self.is_floating_model else 'uint8'})")

    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalise the frame to match model input."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height))
        input_data = np.expand_dims(resized, axis=0)  # (1, H, W, 3)
        if self.is_floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        return input_data

    def process_frame(self, frame: np.ndarray):
        """
        Run TFLite inference on a single BGR frame.

        Returns:
            annotated_frame : frame with bounding-box overlays (BGR np.ndarray)
            detected_labels : list[str] of class names above confidence threshold
            elapsed_ms      : float – inference time in milliseconds
            fps             : float – frames-per-second estimate
        """
        h_frame, w_frame = frame.shape[:2]
        input_data = self._preprocess(frame)

        # ── Inference ─────────────────────────────────────────────────────────
        start = time.time()
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        elapsed_ms = (time.time() - start) * 1000.0
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

        # ── Parse outputs  ────────────────────────────────────────────────────
        # Standard SSD / EfficientDet-Lite output order:
        #   [0] boxes      → (1, N, 4)  [ymin, xmin, ymax, xmax]  (normalised)
        #   [1] classes    → (1, N)
        #   [2] scores     → (1, N)
        #   [3] num_dets   → (1,)
        boxes   = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores  = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        try:
            num_dets = int(self.interpreter.get_tensor(self.output_details[3]["index"])[0])
        except IndexError:
            num_dets = len(scores)

        annotated_frame = frame.copy()
        detected_labels: list[str] = []

        for i in range(num_dets):
            score = float(scores[i])
            if score < self.conf_thres:
                continue

            class_id = int(classes[i])
            label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
            detected_labels.append(label)

            # Un-normalise box coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * w_frame)
            y1 = int(ymin * h_frame)
            x2 = int(xmax * w_frame)
            y2 = int(ymax * h_frame)

            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background + text
            text = f"{label}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated_frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        return annotated_frame, detected_labels, elapsed_ms, fps
