import cv2

class CameraModule:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def initialize(self):
        """Starts the camera feed."""
        print(f"[CAMERA] Initializing camera at index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # In OpenCV, some backends might not support setting resolution directly,
        # but we attempt it here.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open camera {self.camera_index}")
            return False
        print(f"[CAMERA] Successfully opened camera {self.camera_index}.")
        return True

    def get_frame(self):
        """Reads a frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Releases the camera resources."""
        if self.cap is not None:
            self.cap.release()
            print("[CAMERA] Released.")
