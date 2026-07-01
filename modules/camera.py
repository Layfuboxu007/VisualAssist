import cv2

class CameraModule:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def initialize(self):
        """Starts the camera feed."""
        import sys
        
        # We will try backends to see which one can open and successfully read a frame
        backends = [None]  # None means default backend
        if sys.platform.startswith('win32'):
            backends.append(cv2.CAP_DSHOW)
            
        for backend in backends:
            backend_name = "default" if backend is None else "CAP_DSHOW"
            print(f"[CAMERA] Trying to initialize camera at index {self.camera_index} with backend: {backend_name}...")
            
            if backend is None:
                cap = cv2.VideoCapture(self.camera_index)
            else:
                cap = cv2.VideoCapture(self.camera_index, backend)
                
            if not cap.isOpened():
                print(f"[CAMERA] Failed to open camera with backend {backend_name}.")
                cap.release()
                continue
                
            # Set resolution properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Test frame read to ensure backend is actually working
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[CAMERA] Backend {backend_name} opened but failed to grab test frame.")
                cap.release()
                continue
                
            # Success!
            self.cap = cap
            print(f"[CAMERA] Successfully opened and verified camera {self.camera_index} using backend {backend_name}.")
            return True
            
        print(f"[ERROR] All backend attempts failed to initialize camera {self.camera_index}.")
        return False

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
