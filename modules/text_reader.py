import cv2
import threading
import time
import easyocr

class TextReader:
    def __init__(self):
        print("[OCR] Initializing EasyOCR Reader (English)...")
        try:
            # easyocr automatically handles CPU/GPU logic internally
            self.reader = easyocr.Reader(['en'], verbose=False)
            print("[OCR] EasyOCR Reader initialized successfully.")
        except Exception as e:
            print(f"[OCR ERROR] Failed to initialize EasyOCR: {e}")
            self.reader = None
            
        self.is_processing = False
        self._lock = threading.Lock()
        
        # State tracking for continuous silent reading mode
        self.last_read_text = ""
        self.last_read_time = 0.0

    def read_text_async(self, frame, audio, logger=None, is_silent=False):
        """
        Spawns a background thread to process the frame using EasyOCR.
        Avoids blocking the main application video loop.
        """
        with self._lock:
            if self.is_processing:
                return False
            self.is_processing = True

        # Let the user know the process has started (only if not silent/continuous)
        if not is_silent:
            audio.speak("Reading text")
        
        # Run in thread
        t = threading.Thread(target=self._ocr_worker, args=(frame.copy(), audio, logger, is_silent))
        t.daemon = True
        t.start()
        return True

    def _ocr_worker(self, frame, audio, logger, is_silent):
        try:
            if not self.reader:
                if not is_silent:
                    audio.speak("O C R module not available.")
                return

            h, w = frame.shape[:2]
            
            # Crop the central 60% of the image to target what the user is pointing at
            crop_w = int(w * 0.6)
            crop_h = int(h * 0.6)
            x1 = (w - crop_w) // 2
            y1 = (h - crop_h) // 2
            roi = frame[y1:y1+crop_h, x1:x1+crop_w]
            
            # Convert to grayscale to improve contrast
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Run OCR
            start_time = time.time()
            results = self.reader.readtext(gray)
            duration_ms = (time.time() - start_time) * 1000.0
            
            # Extract and join detected text blocks
            text_blocks = [res[1] for res in results if res[2] > 0.3]  # filter confidence > 0.3
            detected_text = " ".join(text_blocks).strip()
            
            current_time = time.time()
            if detected_text:
                # If silent continuous mode, prevent repeating the same text too quickly
                if is_silent:
                    if detected_text == self.last_read_text and (current_time - self.last_read_time < 10.0):
                        return
                
                self.last_read_text = detected_text
                self.last_read_time = current_time

                # Truncate text if it is extremely long for speech
                speech_text = detected_text
                if len(speech_text) > 150:
                    speech_text = speech_text[:147] + "..."
                    
                print(f"[OCR RESULT] Extracted text ({duration_ms:.1f}ms): {detected_text}")
                audio.speak(f"The text says: {speech_text}")
                
                if logger:
                    logger.log_event("ocr", {
                        "text": detected_text,
                        "duration_ms": duration_ms
                    })
            else:
                if not is_silent:
                    print(f"[OCR RESULT] No text detected ({duration_ms:.1f}ms).")
                    audio.speak("No text detected")
                
        except Exception as e:
            print(f"[OCR ERROR] Worker thread failed: {e}")
            if not is_silent:
                audio.speak("O C R processing failed.")
        finally:
            with self._lock:
                self.is_processing = False
