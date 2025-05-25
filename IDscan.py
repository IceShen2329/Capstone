import cv2
import pytesseract
import re
import numpy as np
from PIL import Image
import os
from datetime import datetime

class IDScanner:
    def __init__(self):
        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Configure Tesseract (adjust path if needed)
        # For Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        self.last_scanned_data = {"student_no": "", "name": "", "course": "", "year": ""}
        
        # Create directory for saved screenshots
        self.screenshot_dir = "id_screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
        
        # Button state tracking
        self.button_pressed = False
        self.button_hover = False
        
    def calculate_scan_area(self, frame_width, frame_height):
        """Calculate optimal scan area based on frame resolution and ID card proportions"""
        # Modified aspect ratio to make scanning area shorter (reduced height)
        # Standard ID card is 1.586:1, but we'll use a wider ratio to reduce height
        id_aspect_ratio = 2.2  # Increased from 1.586 to make it shorter/wider
        
        # Calculate scan area dimensions as percentage of frame size
        # Reduced width ratios for narrower scanning area
        if frame_width >= 1920:  # 1080p and higher
            scan_width_ratio = 0.35  # Reduced from 0.5 to 0.35 (35% of frame width)
            margin_ratio = 0.05      # 5% margin from edges
        elif frame_width >= 1280:  # 720p
            scan_width_ratio = 0.4   # Reduced from 0.6 to 0.4 (40% of frame width)
            margin_ratio = 0.08      # 8% margin from edges
        else:  # Lower resolutions
            scan_width_ratio = 0.5   # Reduced from 0.7 to 0.5 (50% of frame width)
            margin_ratio = 0.1       # 10% margin from edges
        
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
    
    def calculate_button_area(self, frame_width, frame_height, scan_x, scan_y, scan_width, scan_height):
        """Calculate the scan button position and dimensions"""
        # Button dimensions scaled by resolution
        if frame_width >= 1920:
            button_width = 200
            button_height = 60
        elif frame_width >= 1280:
            button_width = 160
            button_height = 50
        else:
            button_width = 120
            button_height = 40
        
        # Position button below the scan area with some margin
        button_x = scan_x + (scan_width - button_width) // 2  # Center horizontally with scan area
        button_y = scan_y + scan_height + 30  # 30 pixels below scan area
        
        # Ensure button doesn't go off screen
        if button_y + button_height > frame_height - 20:
            button_y = scan_y - button_height - 30  # Place above scan area if no room below
        
        return button_x, button_y, button_width, button_height
    
    def draw_scan_button(self, frame, button_x, button_y, button_width, button_height):
        """Draw an interactive scan button"""
        # Button colors based on state
        if self.button_pressed:
            button_color = (0, 150, 0)  # Darker green when pressed
            text_color = (255, 255, 255)
            border_color = (0, 100, 0)
        elif self.button_hover:
            button_color = (50, 255, 50)  # Lighter green on hover
            text_color = (0, 0, 0)
            border_color = (0, 200, 0)
        else:
            button_color = (0, 200, 0)  # Normal green
            text_color = (255, 255, 255)
            border_color = (0, 255, 0)
        
        # Draw button background
        cv2.rectangle(frame, (button_x, button_y), 
                     (button_x + button_width, button_y + button_height), 
                     button_color, -1)
        
        # Draw button border
        cv2.rectangle(frame, (button_x, button_y), 
                     (button_x + button_width, button_y + button_height), 
                     border_color, 3)
        
        # Add button text
        font_scale = button_width / 200 * 0.7  # Scale font with button size
        text = "SCAN ID"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = button_x + (button_width - text_size[0]) // 2
        text_y = button_y + (button_height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
        
        # Add scan icon (simple camera-like shape)
        icon_size = min(button_height // 3, 15)
        icon_x = button_x + 15
        icon_y = button_y + (button_height - icon_size) // 2
        
        # Draw simple camera icon
        cv2.rectangle(frame, (icon_x, icon_y), 
                     (icon_x + icon_size, icon_y + icon_size), 
                     text_color, 2)
        cv2.circle(frame, (icon_x + icon_size//2, icon_y + icon_size//2), 
                  icon_size//3, text_color, -1)
        
        return button_x, button_y, button_width, button_height
    
    def is_point_in_button(self, x, y, button_x, button_y, button_width, button_height):
        """Check if a point is inside the button area"""
        return (button_x <= x <= button_x + button_width and 
                button_y <= y <= button_y + button_height)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for button interaction"""
        if hasattr(self, 'button_area'):
            button_x, button_y, button_width, button_height = self.button_area
            
            # Check if mouse is over button
            self.button_hover = self.is_point_in_button(x, y, button_x, button_y, button_width, button_height)
            
            # Handle button click
            if event == cv2.EVENT_LBUTTONDOWN and self.button_hover:
                self.button_pressed = True
            elif event == cv2.EVENT_LBUTTONUP:
                if self.button_pressed and self.button_hover:
                    # Button was clicked - trigger scan
                    self.trigger_scan = True
                self.button_pressed = False
    
    def draw_scan_overlay(self, frame):
        """Draw scanning overlay with guidelines and instructions"""
        height, width = frame.shape[:2]
        x, y, scan_width, scan_height = self.calculate_scan_area(width, height)
        
        # Draw main scanning rectangle
        cv2.rectangle(frame, (x, y), (x + scan_width, y + scan_height), (0, 255, 0), 3)
        
        # Draw corner indicators for better visual guidance
        corner_length = min(30, scan_width // 10, scan_height // 10)
        thickness = 4
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_length, y), (0, 255, 0), thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), (0, 255, 0), thickness)
        
        # Top-right corner
        cv2.line(frame, (x + scan_width, y), (x + scan_width - corner_length, y), (0, 255, 0), thickness)
        cv2.line(frame, (x + scan_width, y), (x + scan_width, y + corner_length), (0, 255, 0), thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x, y + scan_height), (x + corner_length, y + scan_height), (0, 255, 0), thickness)
        cv2.line(frame, (x, y + scan_height), (x, y + scan_height - corner_length), (0, 255, 0), thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x + scan_width, y + scan_height), (x + scan_width - corner_length, y + scan_height), (0, 255, 0), thickness)
        cv2.line(frame, (x + scan_width, y + scan_height), (x + scan_width, y + scan_height - corner_length), (0, 255, 0), thickness)
        
        # Draw center crosshair for alignment
        center_x = x + scan_width // 2
        center_y = y + scan_height // 2
        crosshair_size = 20
        cv2.line(frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), (0, 255, 0), 2)
        
        # Calculate and draw the scan button
        button_x, button_y, button_width, button_height = self.calculate_button_area(width, height, x, y, scan_width, scan_height)
        self.button_area = (button_x, button_y, button_width, button_height)
        self.draw_scan_button(frame, button_x, button_y, button_width, button_height)
        
        # Add instruction text with better positioning
        font_scale = width / 1920 * 0.7  # Scale font with resolution
        instruction_text = "Position ID card in frame - Click SCAN ID button or press SPACE - Press 'q' to quit"
        text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = (width - text_size[0]) // 2
        
        # Position text above scan area or at top if no room
        text_y = y - 30 if y > 60 else 30
        
        # Add background for better text visibility
        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                     (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, instruction_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        
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
    
    def extract_student_info(self, text):
        """Extract student number, name, course, and year from OCR text"""
        print(f"Raw OCR Text:\n{text}")  # Debug: Show raw OCR output
        print("-" * 40)
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]  # Remove empty lines
        student_no = ""
        name = ""
        course = ""
        year = ""
        
        # Look for student number pattern (like 1284-21)
        student_no_pattern = r'\b\d{4}-\d{2}\b'
        
        # Look for course patterns (like BSCPE, BSIT, etc.)
        course_pattern = r'\b(BS[A-Z]{2,4}|BA[A-Z]{2,4}|AB[A-Z]{2,4}|BSC[A-Z]{2,4})\b'
        
        # Look for year patterns
        year_pattern = r'\b(FIRST|SECOND|THIRD|FOURTH|1ST|2ND|3RD|4TH)[\s]*YEAR\b'
        
        name_found = False
        
        print("Processing lines:")
        for i, line in enumerate(lines):
            print(f"Line {i}: '{line}'")
            line_upper = line.upper()
            
            # Search for student number
            student_match = re.search(student_no_pattern, line)
            if student_match:
                student_no = student_match.group()
                print(f"  -> Found student number: {student_no}")
            
            # Search for course
            course_match = re.search(course_pattern, line_upper)
            if course_match:
                course = course_match.group()
                print(f"  -> Found course: {course}")
            
            # Search for year
            year_match = re.search(year_pattern, line_upper)
            if year_match:
                year = year_match.group()
                print(f"  -> Found year: {year}")
            
            # Enhanced name search - Multiple approaches
            if not name_found:
                # Method 1: Look for "NAME" keyword patterns
                name_keywords = ["NAME", "STUDENT NAME", "FULL NAME", "NOMBRE"]
                for keyword in name_keywords:
                    if keyword in line_upper:
                        # Try same line after keyword
                        if ":" in line:
                            name_part = line.split(":", 1)
                            if len(name_part) > 1:
                                potential_name = name_part[1].strip()
                                if len(potential_name) > 2:
                                    name = potential_name
                                    name_found = True
                                    print(f"  -> Found name (same line with :): {name}")
                                    break
                        
                        # Try next line
                        if not name_found and i + 1 < len(lines):
                            potential_name = lines[i + 1].strip()
                            if len(potential_name) > 2 and not re.search(r'\d{4}-\d{2}', potential_name):
                                name = potential_name
                                name_found = True
                                print(f"  -> Found name (next line): {name}")
                                break
                        
                        # Try removing keyword from same line
                        if not name_found:
                            clean_line = line_upper.replace(keyword, "").strip()
                            if len(clean_line) > 2:
                                name = clean_line.title()  # Convert to proper case
                                name_found = True
                                print(f"  -> Found name (cleaned same line): {name}")
                                break
                
                # Method 2: Look for lines that look like names (multiple words, mostly letters)
                if not name_found and len(line) > 3:
                    # Check if line looks like a name
                    # Remove common punctuation and check
                    clean_line = re.sub(r'[^\w\s]', ' ', line).strip()
                    words = clean_line.split()
                    
                    if (len(words) >= 2 and 
                        len(clean_line) >= 5 and
                        len(clean_line) <= 50 and  # Names shouldn't be too long
                        not re.search(student_no_pattern, line) and  # Not a student number
                        not re.search(course_pattern, line_upper) and  # Not a course
                        "YEAR" not in line_upper and  # Not a year
                        not re.search(r'\d{4}', line) and  # No 4-digit numbers (years/IDs)
                        sum(1 for c in line if c.isalpha()) / len(line) > 0.7):  # Mostly letters
                        
                        # Additional checks for name-like patterns
                        if (any(word[0].isupper() for word in words) or  # Has capitalized words
                            any(len(word) > 2 for word in words)):  # Has substantial words
                            name = line.strip()
                            name_found = True
                            print(f"  -> Found name (pattern matching): {name}")
                
                # Method 3: Look for common name positions (usually after student number)
                if not name_found and student_no and i > 0:
                    # Check if this line could be a name following student number
                    prev_line = lines[i-1] if i > 0 else ""
                    if (student_no in prev_line and 
                        len(line) > 3 and 
                        not re.search(r'\d', line) and  # No numbers
                        line.replace(' ', '').replace('.', '').replace(',', '').isalpha()):
                        name = line.strip()
                        name_found = True
                        print(f"  -> Found name (after student number): {name}")
        
        print("-" * 40)
        return student_no, name, course, year
    
    def capture_and_process_scan(self, frame, scan_area):
        """Capture the scan area, save it as image, and extract information"""
        x, y, w, h = scan_area
        
        # Extract the scan area from the frame
        scan_region = frame[y:y+h, x:x+w]
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the original color scan area
        original_filename = os.path.join(self.screenshot_dir, f"id_scan_{timestamp}_original.jpg")
        cv2.imwrite(original_filename, scan_region)
        
        # Preprocess the image for OCR
        processed = self.preprocess_image(scan_region)
        
        # Save the processed image too
        processed_filename = os.path.join(self.screenshot_dir, f"id_scan_{timestamp}_processed.jpg")
        cv2.imwrite(processed_filename, processed)
        
        # Convert to PIL Image for pytesseract
        pil_image = Image.fromarray(processed)
        
        # Perform OCR
        try:
            text = pytesseract.image_to_string(pil_image, config='--psm 6')
            student_no, name, course, year = self.extract_student_info(text)
            
            # Display results in terminal
            print("\n" + "="*60)
            print("SCAN CAPTURED AND PROCESSED!")
            print(f"Files saved:")
            print(f"  - Original: {original_filename}")
            print(f"  - Processed: {processed_filename}")
            print("-" * 60)
            print("EXTRACTED INFORMATION:")
            print(f"STUDENT NO: {student_no if student_no else 'Not found'}")
            print(f"NAME: {name if name else 'Not found'}")
            print(f"COURSE: {course if course else 'Not found'}")
            print(f"YEAR: {year if year else 'Not found'}")
            print("="*60)
            
            # Save to text file as well
            text_filename = os.path.join(self.screenshot_dir, f"id_scan_{timestamp}_data.txt")
            with open(text_filename, "w") as f:
                f.write(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Original Image: {original_filename}\n")
                f.write(f"Processed Image: {processed_filename}\n")
                f.write(f"STUDENT NO: {student_no}\n")
                f.write(f"NAME: {name}\n")
                f.write(f"COURSE: {course}\n")
                f.write(f"YEAR: {year}\n")
                f.write(f"\nRaw OCR Text:\n{text}\n")
            
            print(f"Data also saved to: {text_filename}")
            print("")
            
            return True
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return False
    
    def run(self):
        """Main loop for live feed scanning"""
        print("ID Scanner Started!")
        print("Instructions:")
        print("- Position ID card within the green frame")
        print("- Click the 'SCAN ID' button or press SPACEBAR to capture and process")
        print("- Press 'q' to quit")
        print("- Screenshots will be saved in 'id_screenshots' folder")
        print("-" * 60)
        
        # Initialize mouse callback and trigger flag
        self.trigger_scan = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Create a copy for processing
            display_frame = frame.copy()
            
            # Draw scan overlay and get scan area coordinates
            x, y, scan_width, scan_height = self.draw_scan_overlay(display_frame)
            scan_area = (x, y, scan_width, scan_height)
            
            # Display the frame
            window_name = "ID Scanner - Live Feed"
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            cv2.imshow(window_name, display_frame)   
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') or self.trigger_scan:  # Spacebar or button click
                if self.trigger_scan:
                    print("Button clicked - Capturing scan area...")
                    self.trigger_scan = False  # Reset flag
                else:
                    print("Spacebar pressed - Capturing scan area...")
                self.capture_and_process_scan(frame, scan_area)
         
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'cap'):
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