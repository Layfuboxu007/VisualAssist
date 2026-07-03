import os
import cv2
import pickle
import numpy as np
import torch

class FaceRecognizer:
    def __init__(self, db_path="models/faces.pkl"):
        self.db_path = db_path
        self.faces_db = {}
        
        # Ensure the models directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.load_database()

        # Initialize Haar Cascade face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("[FACE ID ERROR] Could not load Haar Cascade face detector.")

        # Initialize FaceNet embedding extractor
        print("[FACE ID] Initializing FaceNet embedding model...")
        try:
            from facenet_pytorch import InceptionResnetV1
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
            print("[FACE ID] FaceNet model loaded successfully.")
            self.is_ready = True
        except Exception as e:
            print(f"[FACE ID WARNING] Failed to load FaceNet model (perhaps offline): {e}")
            print("[FACE ID] Operating in face-detection fallback mode only.")
            self.resnet = None
            self.is_ready = False

    def load_database(self):
        """Loads registered faces from a pickle file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    self.faces_db = pickle.load(f)
                print(f"[FACE ID] Loaded {len(self.faces_db)} registered face(s) from database.")
            except Exception as e:
                print(f"[FACE ID ERROR] Failed to load face database: {e}")
                self.faces_db = {}
        else:
            self.faces_db = {}
            print("[FACE ID] Database empty. No registered faces found.")

    def save_database(self):
        """Saves registered faces database to pickle file."""
        try:
            with open(self.db_path, "wb") as f:
                pickle.dump(self.faces_db, f)
            print(f"[FACE ID] Saved database with {len(self.faces_db)} registered face(s).")
        except Exception as e:
            print(f"[FACE ID ERROR] Failed to save face database: {e}")

    def _get_embedding(self, face_crop):
        """Extracts 512-dimensional embedding vector from a BGR face crop."""
        if self.resnet is None:
            return None
            
        try:
            # Preprocess: convert BGR to RGB, resize to 160x160
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (160, 160))
            
            # Convert to float32 and normalize (img - 127.5) / 128.0
            face_img = np.float32(resized)
            face_img = (face_img - 127.5) / 128.0
            
            # Reshape to (C, H, W) and add batch dimension (1, C, H, W)
            face_tensor = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.resnet(face_tensor).squeeze(0).numpy()
            return embedding
        except Exception as e:
            print(f"[FACE ID ERROR] Embedding extraction failed: {e}")
            return None

    def register_face(self, frame, bbox, name):
        """
        Extracts face embedding from a bounding box and registers it under a name.
        Returns:
            bool: True if registration was successful, False otherwise.
        """
        x, y, w, h = bbox
        # Avoid out of bounds crop
        img_h, img_w = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            print("[FACE ID ERROR] Empty face crop for registration.")
            return False

        embedding = self._get_embedding(face_crop)
        if embedding is None:
            # Fallback registration: we store a dummy embedding if model isn't active
            # (or we can block registration if embedding failed)
            if not self.is_ready:
                # Store a mock signature (average color histogram of face ROI as a stub)
                hist = cv2.calcHist([face_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                embedding = cv2.normalize(hist, hist).flatten()
            else:
                return False

        self.faces_db[name] = embedding
        self.save_database()
        print(f"[FACE ID] Successfully registered face: {name}")
        return True

    def recognize_faces(self, frame):
        """
        Detects faces in the frame and attempts to recognize them.
        Returns:
            list of dict: [{"box": [x, y, w, h], "name": name, "confidence": float}]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optimization: Downscale grayscale image by 2x for fast Haar face detection (4x speedup)
        scale = 0.5
        small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        # Detect faces with multiscale Haar cascade on downscaled frame
        faces = self.face_cascade.detectMultiScale(
            small_gray, 
            scaleFactor=1.2, 
            minNeighbors=5, 
            minSize=(25, 25)
        )

        detections = []
        for (x, y, w, h) in faces:
            # Scale coordinates back to original size
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_w = int(w / scale)
            orig_h = int(h / scale)
            
            # Crop face region from original high-res frame
            face_crop = frame[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
            if face_crop.size == 0:
                continue

            name = "Unknown"
            max_sim = 0.0

            # Only attempt recognition if model is loaded and we have registered faces
            if self.faces_db:
                embedding = self._get_embedding(face_crop)
                if embedding is not None:
                    # If it's a real model embedding (512-dim) vs mock hist (512-dim flattened)
                    for reg_name, reg_emb in self.faces_db.items():
                        if len(embedding) == len(reg_emb):
                            # Cosine similarity
                            dot_prod = np.dot(embedding, reg_emb)
                            norm_a = np.linalg.norm(embedding)
                            norm_b = np.linalg.norm(reg_emb)
                            if norm_a > 0 and norm_b > 0:
                                sim = dot_prod / (norm_a * norm_b)
                                if sim > max_sim:
                                    max_sim = sim
                                    name = reg_name
                    
                    # Threshold for face match
                    threshold = 0.58
                    if max_sim < threshold:
                        name = "Unknown"
                else:
                    # FaceNet model not loaded, do color histogram matching fallback
                    if not self.is_ready:
                        hist = cv2.calcHist([face_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        curr_emb = cv2.normalize(hist, hist).flatten()
                        for reg_name, reg_emb in self.faces_db.items():
                            if len(curr_emb) == len(reg_emb):
                                # Histogram correlation
                                sim = cv2.compareHist(
                                    curr_emb.reshape(8, 8, 8).astype(np.float32), 
                                    reg_emb.reshape(8, 8, 8).astype(np.float32), 
                                    cv2.HISTCMP_CORREL
                                )
                                if sim > max_sim:
                                    max_sim = sim
                                    name = reg_name
                        threshold = 0.70
                        if max_sim < threshold:
                            name = "Unknown"

            detections.append({
                "box": [orig_x, orig_y, orig_w, orig_h],
                "name": name,
                "confidence": float(max_sim)
            })

        return detections
