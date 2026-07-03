import os
import cv2
import sys

# Append parent directory to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.face_id import FaceRecognizer

def main():
    # Initialize the FaceRecognizer with the same models/faces.pkl database
    face_id = FaceRecognizer(db_path="models/faces.pkl")
    
    # Create the directory for images if it doesn't exist
    img_dir = "register_faces"
    os.makedirs(img_dir, exist_ok=True)
    
    print("==================================================")
    print("      SightAssist: Face Pre-Registration Utility  ")
    print("==================================================")
    print(f"Instructions:")
    print(f"1. Place clear face photos (JPG, PNG) in the '{img_dir}' folder.")
    print(f"2. Name each image file with the person's name (e.g., 'Ahmad.jpg', 'Dr_Smith.png').")
    print(f"3. Run this script to batch register them into the database.")
    print("--------------------------------------------------")
    
    # Read files in the register_faces directory
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No images found in '{img_dir}' directory yet.")
        print(f"Please place your photo files in the folder '{img_dir}/' and run this script.")
        return
        
    success_count = 0
    for file in files:
        # File name (excluding extension) is used as the person's registered name
        name = os.path.splitext(file)[0].strip()
        img_path = os.path.join(img_dir, file)
        print(f"Processing face for '{name}' from image: {file}...")
        
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[ERROR] Could not load image file: {file}")
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Multi-scale face detection on grayscale image
        faces = face_id.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            print(f"[WARNING] Face detector did not find a face in: {file}. Skipping.")
            continue
            
        # Select the largest face in the photo
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Register in pickle database
        success = face_id.register_face(frame, (x, y, w, h), name)
        if success:
            print(f"[SUCCESS] Registered '{name}' successfully.")
            success_count += 1
        else:
            print(f"[ERROR] Failed to register face embedding for '{name}'.")
            
    print("--------------------------------------------------")
    print(f"Process complete. Registered {success_count}/{len(files)} face(s) in database.")
    print("==================================================")

if __name__ == "__main__":
    main()
