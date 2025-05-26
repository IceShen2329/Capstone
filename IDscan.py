import cv2
import pytesseract
import re
import numpy as np
from datetime import datetime
import os
import sys
import subprocess
import pygame
import time

class IDScanner:
    def __init__(self):
        self.cap = None
        self.camera_index = 0
        self.last_scanned_data = {"student_no": "", "name": ""}
        
        # Create directory for saved text files
        self.text_output_dir = "id_text_output"
        if not os.path.exists(self.text_output_dir):
            os.makedirs(self.text_output_dir)
        
        # Auto-scanning variables
        self.last_scan_time = 0
        self.scan_interval = 5.0  # Changed to 5 seconds
        self.scanning_active = True
        self.current_scan_data = {"student_no": "", "name": ""}
        
        # ID detection tracking
        self.last_id_detection_time = 0
        self.id_detection_timeout = 3.0  # Reset data if no ID detected for 3 seconds
        
        # Initialize camera with better error handling
        self.initialize_camera()
    
    def find_available_cameras(self):
        """Find all available camera indices"""
        available_cameras = []
        for i in range(1):  # Check first 5 camera indices
            cap = cv2.VideoCapture(1)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(1)
                    print(f"Found working camera at index {1}")
                cap.release()
            time.sleep(0.1)  # Small delay between attempts
        return available_cameras
    
    def initialize_camera(self):
        """Initialize camera with multiple fallback options"""
        print("Initializing camera...")
        
        # Find available cameras
        available_cameras = self.find_available_cameras()
        
        if not available_cameras:
            print("No cameras found!")
            return False
        
        # Try each available camera
        for cam_index in available_cameras:
            print(f"Trying camera index {cam_index}...")
            
            # Try different backends
            backends = [
                cv2.CAP_DSHOW,      # DirectShow (Windows)
                cv2.CAP_MSMF,       # Microsoft Media Foundation
                cv2.CAP_V4L2,       # Video4Linux (Linux)
                cv2.CAP_ANY         # Auto-detect
            ]
            
            for backend in backends:
                try:
                    if self.cap is not None:
                        self.cap.release()
                        time.sleep(0.2)
                    
                    print(f"  Trying backend: {backend}")
                    self.cap = cv2.VideoCapture(cam_index, backend)
                    
                    if not self.cap.isOpened():
                        continue
                    
                    # Set buffer size to reduce latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test if we can actually read frames
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"✓ Camera {cam_index} working with backend {backend}")
                        
                        # Try to set resolution (don't fail if it doesn't work)
                        try:
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        except:
                            pass
                        
                        self.camera_index = cam_index
                        return True
                        
                except Exception as e:
                    print(f"  Backend {backend} failed: {e}")
                    continue
        
        print("❌ Failed to initialize any camera")
        return False
    
    def reconnect_camera(self):
        """Attempt to reconnect the camera"""
        print("Attempting to reconnect camera...")
        if self.cap is not None:
            self.cap.release()
            time.sleep(1)  # Wait a bit before reconnecting
        
        return self.initialize_camera()
    
    def safe_read_frame(self):
        """Safely read a frame with error handling"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            if self.cap is None or not self.cap.isOpened():
                print("Camera not opened, attempting to reconnect...")
                if not self.reconnect_camera():
                    return False, None
            
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True, frame
                else:
                    print(f"Failed to read frame (attempt {retry_count + 1})")
                    retry_count += 1
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Exception reading frame: {e}")
                retry_count += 1
                time.sleep(0.1)
        
        # If all retries failed, try to reconnect
        print("All frame read attempts failed, trying to reconnect camera...")
        if self.reconnect_camera():
            try:
                return self.cap.read()
            except:
                return False, None
        
        return False, None
    
    def save_temp_scan_data(self, student_no, name):
        """Save scan data to temp file for confirmation screen"""
        try:
            # Save to GUI folder (parent directory)
            temp_file_path = os.path.join(os.path.dirname(__file__), "..", "GUI", "temp_scan_data.txt")
            temp_file_path = os.path.normpath(temp_file_path)
            
            # Create GUI directory if it doesn't exist
            gui_dir = os.path.dirname(temp_file_path)
            if not os.path.exists(gui_dir):
                os.makedirs(gui_dir)
                
            with open(temp_file_path, "w") as f:
                f.write(f"{student_no}\n")
                f.write(f"{name}\n")
            
            print(f"Temp data saved to: {temp_file_path}")
            return True
        except Exception as e:
            print(f"Error saving temp data: {e}")
            return False
    
    def launch_confirmation(self):
        """Launch the confirmation GUI"""
        try:
            # Clean up camera and CV2 windows
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Path to confirmation.py in GUI folder
            confirmation_script = os.path.join(os.path.dirname(__file__), "..", "GUI", "confirmation.py")
            confirmation_script = os.path.normpath(confirmation_script)
            
            print(f"Launching confirmation script: {confirmation_script}")
            subprocess.Popen([sys.executable, confirmation_script])
            sys.exit()
        except Exception as e:
            print(f"Error launching confirmation: {e}")
            sys.exit(1)
    
    def calculate_scan_area(self, frame_width, frame_height):
        """Calculate optimal scan area based on frame resolution and ID card proportions"""
        # Modified aspect ratio to make scanning area shorter (reduced height)
        id_aspect_ratio = 2.2  # Increased from 1.586 to make it shorter/wider
        
        # Calculate scan area dimensions as percentage of frame size
        if frame_width >= 1920:  # 1080p and higher
            scan_width_ratio = 0.35
            margin_ratio = 0.05
        elif frame_width >= 1280:  # 720p
            scan_width_ratio = 0.4
            margin_ratio = 0.08
        else:  # Lower resolutions
            scan_width_ratio = 0.5
            margin_ratio = 0.1
        
        # Calculate scan area dimensions
        scan_width = int(frame_width * scan_width_ratio)
        scan_height = int(scan_width / id_aspect_ratio)
        
        # Ensure scan area fits within frame with margins
        max_scan_height = int(frame_height * (1 - 2 * margin_ratio))
        if scan_height > max_scan_height:
            scan_height = max_scan_height
            scan_width = int(scan_height * id_aspect_ratio)
        
        # Center the scan area
        x = (frame_width - scan_width) // 2
        y = (frame_height - scan_height) // 2
        
        return x, y, scan_width, scan_height
    
    def draw_scan_status(self, frame, frame_width, frame_height):
        """Draw scanning status and current scan data"""
        font_scale = frame_width / 1920 * 0.7
        
        # Status indicator
        current_time = time.time()
        time_since_scan = current_time - self.last_scan_time
        
        if time_since_scan < 0.5:  # Flash green when scanning
            status_color = (0, 255, 0)
            status_text = "SCANNING..."
        elif self.all_fields_found():
            status_color = (0, 255, 0)
            status_text = "ID DETECTED - COMPLETE"
        else:
            status_color = (0, 255, 255)
            status_text = "LOOKING FOR ID..."
        
        # Draw status
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        status_x = (frame_width - status_size[0]) // 2
        status_y = 50
        
        cv2.rectangle(frame, (status_x - 10, status_y - status_size[1] - 10), 
                     (status_x + status_size[0] + 10, status_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (status_x, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, 2)
        
        # Draw current scan data
        y_offset = 100
        data_font_scale = font_scale * 0.8
        
        fields = [
            ("STUDENT NO:", self.current_scan_data["student_no"]),
            ("NAME:", self.current_scan_data["name"])
        ]
        
        for label, value in fields:
            display_value = value if value else "Not detected"
            color = (0, 255, 0) if value else (0, 255, 255)
            
            text = f"{label} {display_value}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, data_font_scale, 2)[0]
            text_x = 20
            text_y = y_offset
            
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                         (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, data_font_scale, color, 2)
            
            y_offset += 40
    
    def draw_scan_overlay(self, frame):
        """Draw scanning overlay with guidelines and instructions"""
        height, width = frame.shape[:2]
        x, y, scan_width, scan_height = self.calculate_scan_area(width, height)
        
        # Draw main scanning rectangle
        rect_color = (0, 255, 0) if self.all_fields_found() else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + scan_width, y + scan_height), rect_color, 3)
        
        # Draw corner indicators
        corner_length = min(30, scan_width // 10, scan_height // 10)
        thickness = 2
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_length, y), rect_color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), rect_color, thickness)
        
        # Top-right corner
        cv2.line(frame, (x + scan_width, y), (x + scan_width - corner_length, y), rect_color, thickness)
        cv2.line(frame, (x + scan_width, y), (x + scan_width, y + corner_length), rect_color, thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x, y + scan_height), (x + corner_length, y + scan_height), rect_color, thickness)
        cv2.line(frame, (x, y + scan_height), (x, y + scan_height - corner_length), rect_color, thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x + scan_width, y + scan_height), (x + scan_width - corner_length, y + scan_height), rect_color, thickness)
        cv2.line(frame, (x + scan_width, y + scan_height), (x + scan_width, y + scan_height - corner_length), rect_color, thickness)
        
        # Draw center crosshair for alignment
        center_x = x + scan_width // 2
        center_y = y + scan_height // 2
        crosshair_size = 20
        cv2.line(frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), rect_color, 2)
        cv2.line(frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), rect_color, 2)
        
        # Draw scanning status and data
        self.draw_scan_status(frame, width, height)
        
        # Add instruction text
        font_scale = width / 1920 * 0.6
        instruction_text = "Hold ID card steady in frame - Auto-scanning every 5 seconds - Press 'q' to quit"
        text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 30
        
        # Add background for better text visibility
        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                     (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, instruction_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
        
        return x, y, scan_width, scan_height
    
    def preprocess_image(self, image):
        """Preprocess the image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def clean_special_characters(self, text):
        """Remove special characters from text, keeping only letters, spaces, periods, and hyphens"""
        if not text:
            return ""
        
        # Keep only letters, spaces, periods, hyphens, and apostrophes (common in names)
        # Remove other special characters that might be OCR artifacts
        cleaned = re.sub(r'[^a-zA-Z\s.\'-]', '', text)
        
        # Clean up multiple spaces and trim
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove leading/trailing periods or hyphens
        cleaned = cleaned.strip('.-')
        
        return cleaned
    
    def extract_student_info(self, text):
        """Extract student number and name from OCR text"""
        print(f"Raw OCR Text:\n{text}")
        print("-" * 40)
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        student_no = ""
        name = ""
        
        # Look for student number pattern (like 1284-21)
        student_no_pattern = r'\b\d{4}-\d{2}\b'
        
        # Labels to exclude from name detection
        excluded_labels = [
            "STUDENT NO", "STUDENT NO.", "NAME", "COURSE", "YEAR", 
            "LYCEUM", "REPUBLIC", "PHILIPPINES", "ALABANG", "CERTIFIED",
            "SEMESTER", "SCHOOL", "COLLEGE", "ENGINEERING", "BSCPE",
            "THIRD YEAR", "2ND SEMESTER", "2023-2024"
        ]
        
        name_found = False
        
        print("Processing lines:")
        for i, line in enumerate(lines):
            print(f"Line {i}: '{line}'")
            line_upper = line.upper().strip()
            
            # Search for student number
            student_match = re.search(student_no_pattern, line)
            if student_match:
                student_no = student_match.group()
                print(f"  -> Found student number: {student_no}")
            
            # Enhanced name search with special character cleaning
            if not name_found:
                # Method 1: Look for "NAME" keyword and get the next valid line
                if "NAME" in line_upper and not name_found:
                    # Check if there's content after "NAME:" on the same line
                    if ":" in line:
                        name_part = line.split(":", 1)
                        if len(name_part) > 1:
                            potential_name = name_part[1].strip()
                            # Clean special characters from potential name
                            cleaned_name = self.clean_special_characters(potential_name)
                            if self.is_valid_name(cleaned_name, excluded_labels):
                                name = cleaned_name
                                name_found = True
                                print(f"  -> Found name (same line after NAME:): {name}")
                                continue
                    
                    # Look in the next few lines for the actual name
                    for j in range(i + 1, min(i + 4, len(lines))):
                        next_line = lines[j].strip()
                        # Clean special characters from next line
                        cleaned_next_line = self.clean_special_characters(next_line)
                        if self.is_valid_name(cleaned_next_line, excluded_labels):
                            name = cleaned_next_line
                            name_found = True
                            print(f"  -> Found name (line {j} after NAME): {name}")
                            break
                
                # Method 2: Look for lines that appear to be names (after we have student number)
                elif student_no and not name_found:
                    # Clean special characters from line
                    cleaned_line = self.clean_special_characters(line)
                    if self.is_valid_name(cleaned_line, excluded_labels):
                        name = cleaned_line
                        name_found = True
                        print(f"  -> Found name (pattern matching): {name}")
        
        print("-" * 40)
        return student_no, name
    
    def is_valid_name(self, text, excluded_labels):
        """Check if a text line is likely to be a valid name"""
        if not text or len(text.strip()) < 3:
            return False
        
        text_upper = text.upper().strip()
        
        # Exclude obvious labels and non-name content
        for label in excluded_labels:
            if label in text_upper:
                return False
        
        # Exclude lines with numbers (except simple patterns that might be in names)
        if re.search(r'\d{2,}', text):  # Exclude lines with 2+ consecutive digits
            return False
        
        # After cleaning, the text should not contain problematic special characters
        # Check for remaining unwanted characters (OCR artifacts)
        unwanted_chars = ['@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', 
                         '{', '}', '[', ']', '|', '\\', '/', '<', '>', '?', '`', '~']
        if any(char in text for char in unwanted_chars):
            return False
        
        # Must contain mostly alphabetic characters
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.6:
            return False
        
        # Should look like a name (multiple words, proper capitalization)
        words = text.split()
        if len(words) >= 2:
            # Check if it looks like a proper name format
            if any(word[0].isupper() for word in words if word):
                return True
        
        # Single word names are also possible but should be substantial
        if len(words) == 1 and len(text) >= 4:
            return True
            
        return False
    
    def all_fields_found(self):
        """Check if all required fields have been found"""
        return (self.current_scan_data["student_no"] and 
                self.current_scan_data["name"])
    
    def check_and_reset_if_no_id(self):
        """Check if too much time has passed without detecting an ID and reset data"""
        current_time = time.time()
        
        # If we haven't detected an ID recently and we have stored data, reset it
        if (current_time - self.last_id_detection_time > self.id_detection_timeout and
            (self.current_scan_data["student_no"] or self.current_scan_data["name"])):
            
            print("No ID detected for too long - resetting scan data")
            self.current_scan_data = {"student_no": "", "name": ""}
    
    def auto_scan_and_process(self, frame, scan_area):
        """Automatically scan the area and update current scan data"""
        current_time = time.time()
        
        # Check if we should reset data due to no ID detection
        self.check_and_reset_if_no_id()
        
        # Only scan if enough time has passed
        if current_time - self.last_scan_time < self.scan_interval:
            return False
        
        self.last_scan_time = current_time
        
        x, y, w, h = scan_area
        
        # Extract the scan area from the frame
        scan_region = frame[y:y+h, x:x+w]
        
        # Preprocess the image for OCR
        processed = self.preprocess_image(scan_region)
        
        # Perform OCR
        try:
            text = pytesseract.image_to_string(processed, config='--psm 6')
            student_no, name = self.extract_student_info(text)
            
            # Check if we detected any ID information
            if student_no or name:
                self.last_id_detection_time = current_time
                
                # Update current scan data with any found information
                if student_no:
                    self.current_scan_data["student_no"] = student_no
                if name:
                    self.current_scan_data["name"] = name
                
                # Check if all fields are now found
                if self.all_fields_found():
                    print("\n" + "="*60)
                    print("ALL REQUIRED FIELDS DETECTED!")
                    print("-" * 60)
                    print("EXTRACTED INFORMATION:")
                    print(f"STUDENT NO: {self.current_scan_data['student_no']}")
                    print(f"NAME: {self.current_scan_data['name']}")
                    print("="*60)
                    
                    # Save to text file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    text_filename = os.path.join(self.text_output_dir, f"id_scan_{timestamp}.txt")
                    
                    with open(text_filename, "w") as f:
                        f.write(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"STUDENT NO: {self.current_scan_data['student_no']}\n")
                        f.write(f"NAME: {self.current_scan_data['name']}\n")
                        f.write(f"\nRaw OCR Text:\n{text}\n")
                    
                    print(f"Text data saved to: {text_filename}")
                    
                    # Save temp data and launch confirmation
                    if self.save_temp_scan_data(self.current_scan_data["student_no"], 
                                              self.current_scan_data["name"]):
                        print("Launching confirmation screen...")
                        self.launch_confirmation()
                    
                    return True
                else:
                    # Show what we've found so far
                    found_fields = []
                    if self.current_scan_data["student_no"]:
                        found_fields.append("Student No.")
                    if self.current_scan_data["name"]:
                        found_fields.append("Name")
                    
                    if found_fields:
                        print(f"Found fields: {', '.join(found_fields)}")
                    
                    return False
            else:
                # No ID information detected in this scan
                print("No ID information detected in current scan")
                return False
                
        except Exception as e:
            print(f"OCR Error: {e}")
            return False
    
    def run(self):
        """Main loop for live feed scanning"""
        if self.cap is None:
            print("❌ No camera available. Exiting...")
            return
        
        print("✓ Auto ID Scanner Started!")
        print("Instructions:")
        print("- Hold your ID card steady within the green frame")
        print("- The scanner will automatically detect and scan every 5 seconds")
        print("- Data will only be saved when both Student Number and Name are detected")
        print("- Data will be reset if no ID is detected for 3 seconds")
        print("- Press 'q' to quit")
        print("- Text data will be saved in 'id_text_output' folder")
        print("- Special characters will be automatically filtered from names")
        print("-" * 60)
        
        window_name = "Auto ID Scanner - Live Feed"
        
        # Create window
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        consecutive_failures = 0
        max_failures = 10
        
        while True:
            ret, frame = self.safe_read_frame()
            
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"Failed to read frame ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures, exiting...")
                    break
                
                # Show error message
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Camera Error - Attempting Reconnection...", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(window_name, error_frame)
                
                key = cv2.waitKey(1000) & 0xFF  # Wait 1 second
                if key == ord('q'):
                    break
                continue
            
            # Reset failure counter on successful frame read
            consecutive_failures = 0
            
            # Create a copy for processing
            display_frame = frame.copy()
            
            # Draw scan overlay and get scan area coordinates
            x, y, scan_width, scan_height = self.draw_scan_overlay(display_frame)
            scan_area = (x, y, scan_width, scan_height)
            
            # Auto-scan the area
            if self.scanning_active:
                self.auto_scan_and_process(frame, scan_area)
            
            # Display the frame
            cv2.imshow(window_name, display_frame)   
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset current scan data
                self.current_scan_data = {"student_no": "", "name": ""}
                print("Scan data reset. Looking for new ID...")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    try:
        # Create and run the scanner
        scanner = IDScanner()
        scanner.run()
    except KeyboardInterrupt:
        print("\nScanner stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()