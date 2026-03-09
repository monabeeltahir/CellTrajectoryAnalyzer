import cv2
import math
import csv
import os

class AngleDrawer:
    """
    A tool to open an image, allow the user to draw a line with a live preview, 
    calculate its left-to-right tilt angle, and save the result to a CSV file.
    """
    def __init__(self, csv_path="angles.csv"):
        # The constructor no longer takes an image path, just the configuration
        self.csv_path = csv_path
        self.points = []
        self.angle_deg = None
        self.img = None
        self.img_display = None
        self.drawing = False  # State to track if the mouse is currently being dragged
        self.image_path = None

    def _mouse_callback(self, event, x, y, flags, param):
        """Handles mouse events for live drawing and final calculation."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.points = [(x, y)]
            self.img_display = self.img.copy() # Clear any previous lines
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # Live preview while dragging
            if self.drawing:
                temp_img = self.img.copy()
                # Draw a temporary red line
                cv2.line(temp_img, self.points[0], (x, y), (0, 0, 255), 2) 
                self.img_display = temp_img
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            self.drawing = False
            self.points.append((x, y))
            
            # Draw the final green line
            self.img_display = self.img.copy()
            cv2.line(self.img_display, self.points[0], self.points[1], (0, 255, 0), 2)

            # Calculate raw differences
            dx = self.points[1][0] - self.points[0][0]
            dy = self.points[0][1] - self.points[1][1] 
            
            # Prevent math errors if the user clicked without dragging
            if dx == 0 and dy == 0:
                return
            
            # Force left-to-right perspective for tilt angle
            if dx < 0:
                dx = -dx
                dy = -dy
            
            self.angle_deg = math.degrees(math.atan2(dy, dx))
            
            # Display the angle directly on the image
            text = f"Tilt: {self.angle_deg:.2f} deg"
            cv2.putText(self.img_display, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _save_to_csv(self):
        """Saves the calculated angle to a CSV file."""
        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['image_path', 'angle_deg'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "image_path": self.image_path ,
                "angle_deg": self.angle_deg
            })

    def run(self, image_path ):
        """Loads the specified image, opens the GUI, and handles user input."""
        self.image_path  = image_path
        self.img = cv2.imread(self.image_path )
        if self.img is None:
            raise ValueError(f"Could not load image at path: {self.image_path }")
        
        # Reset variables for each new run
        self.img_display = self.img.copy()
        self.points = []
        self.angle_deg = None
        
        cv2.namedWindow("Draw a Line")
        cv2.setMouseCallback("Draw a Line", self._mouse_callback)
        
        print(f"\n--- Loaded: {self.image_path } ---")
        print("INSTRUCTIONS:")
        print("- Click and drag to draw the line.")
        print("- Press 'c' to CLEAR the line and redraw.")
        print("- Press 'Enter' to SAVE the angle and close.")
        print("- Press 'q' to QUIT without saving.")
        
        while True:
            cv2.imshow("Draw a Line", self.img_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Clear the screen and reset the angle
                self.img_display = self.img.copy()
                self.points = []
                self.angle_deg = None
                print("Line cleared. Draw again.")
                
            elif key == 13: # Enter key
                break
                
            elif key == ord('q'): # Quit key
                self.angle_deg = None
                break
                
        cv2.destroyAllWindows()
        
        if self.angle_deg is not None:
            self._save_to_csv()
            print(f"Saved angle: {self.angle_deg:.2f}° to {self.csv_path}")
        else:
            print("Operation cancelled or no valid line drawn. Nothing saved.")
            
        return self.angle_deg

# # ==========================================
# # Example Usage
# # ==========================================
# if __name__ == "__main__":
#     # Initialize the drawer once
#     drawer = AngleDrawer(csv_path='channel_tilts.csv')
    
#     # You can now call .run() in a loop with different images
#     angle1 = drawer.run('GrayImage.png')
#     # angle2 = drawer.run('image_2.jpg')
#     pass