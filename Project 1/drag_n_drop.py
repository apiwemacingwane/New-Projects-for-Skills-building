import cv2
from cvzone.HandTrackingModule import HandDetector
import os
import numpy as np

# ------------------------ SETUP ------------------------

def initialize_camera():
    """Initialize webcam with error handling"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def create_sample_shapes():
    """Create sample colored shapes when no PNG files are available"""
    shapes = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    shape_names = ["Red Circle", "Green Square", "Blue Triangle", "Yellow Diamond", "Magenta Heart"]
    
    for i, (color, name) in enumerate(zip(colors, shape_names)):
        # Create a 100x100 image with alpha channel
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        
        if "Circle" in name:
            cv2.circle(img, (50, 50), 40, (*color, 255), -1)
        elif "Square" in name:
            cv2.rectangle(img, (10, 10), (90, 90), (*color, 255), -1)
        elif "Triangle" in name:
            pts = np.array([[50, 10], [10, 90], [90, 90]], np.int32)
            cv2.fillPoly(img, [pts], (*color, 255))
        elif "Diamond" in name:
            pts = np.array([[50, 10], [90, 50], [50, 90], [10, 50]], np.int32)
            cv2.fillPoly(img, [pts], (*color, 255))
        elif "Heart" in name:
            # Simple heart shape using circles and triangle
            cv2.circle(img, (35, 35), 20, (*color, 255), -1)
            cv2.circle(img, (65, 35), 20, (*color, 255), -1)
            pts = np.array([[20, 45], [80, 45], [50, 85]], np.int32)
            cv2.fillPoly(img, [pts], (*color, 255))
        
        shapes.append({
            "image": img,
            "pos": [100 + i * 150, 100],
            "selected": False,
            "name": name
        })
    
    return shapes

def load_draggable_items(image_folder="./"):
    """Load PNG images with better error handling, create samples if none found"""
    draggable_items = []
    
    if os.path.exists(image_folder):
        png_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        
        start_x, start_y = 100, 100
        offset_between = 200
        
        for idx, file_name in enumerate(png_files):
            img_path = os.path.join(image_folder, file_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Warning: Could not load image '{file_name}'")
                    continue
                
                # Ensure image has alpha channel
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # Add alpha channel if missing
                    alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
                    img = np.concatenate([img, alpha], axis=2)
                elif len(img.shape) == 3 and img.shape[2] == 1:
                    # Grayscale to RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
                    img = np.concatenate([img, alpha], axis=2)
                
                # Resize if too large
                if img.shape[0] > 200 or img.shape[1] > 200:
                    scale = min(200/img.shape[0], 200/img.shape[1])
                    new_w = int(img.shape[1] * scale)
                    new_h = int(img.shape[0] * scale)
                    img = cv2.resize(img, (new_w, new_h))
                
                # Initial positions staggered horizontally
                pos = [start_x + idx * offset_between, start_y]
                draggable_items.append({
                    "image": img,
                    "pos": pos,
                    "selected": False,
                    "name": file_name
                })
                print(f"Loaded: {file_name}")
                
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue
    
    # If no images were loaded, create sample shapes
    if not draggable_items:
        print("No image files found. Creating sample shapes for demonstration...")
        draggable_items = create_sample_shapes()
        print("Created 5 sample shapes: Circle, Square, Triangle, Diamond, Heart")
    
    return draggable_items

# Initialize components
cap = initialize_camera()
if cap is None:
    exit(1)

# Initialize Hand Detector with optimized settings
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Load draggable items (will create samples if no images found)
draggable_items = load_draggable_items()

if not draggable_items:
    print("Failed to create any draggable items.")
    cap.release()
    exit(1)

# Configuration
GRAB_DISTANCE_THRESHOLD = 40
grabbed_idx = -1

# ------------------------ UTILITY FUNCTIONS ------------------------

def overlay_png(background, overlay, x, y):
    """
    Overlay a transparent PNG onto background with improved bounds checking
    """
    if overlay.shape[2] != 4:
        return background
    
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    # Boundary checks
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background

    # Calculate valid regions
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, bg_w)
    y2 = min(y + h, bg_h)

    # Corresponding overlay region
    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    # Extract regions
    bg_region = background[y1:y2, x1:x2]
    overlay_region = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    
    if bg_region.shape[0] == 0 or bg_region.shape[1] == 0:
        return background

    # Normalize alpha channel
    alpha = overlay_region[:, :, 3:4] / 255.0
    alpha_inv = 1.0 - alpha

    # Blend colors
    for c in range(3):
        bg_region[:, :, c] = (
            alpha[:, :, 0] * overlay_region[:, :, c] +
            alpha_inv[:, :, 0] * bg_region[:, :, c]
        )

    return background

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def point_in_rect(point, rect_pos, rect_size):
    """Check if point is inside rectangle"""
    x, y = point
    rx, ry = rect_pos
    rw, rh = rect_size
    return rx <= x <= rx + rw and ry <= y <= ry + rh

# ------------------------ MAIN LOOP ------------------------

print("Hand Tracking Drag-and-Drop Started")
print("Controls:")
print("- Pinch (bring index and middle fingers together) to grab items")
print("- Move while pinching to drag")
print("- Release pinch to drop")
print("- Press ESC to exit")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        # Flip for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        hands, frame_with_hands = detector.findHands(frame, draw=False)

        if hands:
            hand = hands[0]
            lm_list = hand["lmList"]
            
            # Get finger positions (landmarks 8 and 12)
            if len(lm_list) >= 13:
                idx_tip = (lm_list[8][0], lm_list[8][1])
                mid_tip = (lm_list[12][0], lm_list[12][1])

                # Draw finger positions for visual feedback
                cv2.circle(frame, idx_tip, 8, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, mid_tip, 8, (255, 0, 255), cv2.FILLED)
                
                # Draw line between fingers
                cv2.line(frame, idx_tip, mid_tip, (255, 255, 0), 2)

                # Calculate pinch distance
                distance = calculate_distance(idx_tip, mid_tip)
                
                # Display distance for debugging
                cv2.putText(frame, f"Distance: {int(distance)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Handle grabbing logic
                if grabbed_idx != -1:
                    # Currently holding an item
                    if distance < GRAB_DISTANCE_THRESHOLD:
                        # Still pinching - move the item
                        item = draggable_items[grabbed_idx]
                        img_h, img_w = item["image"].shape[:2]
                        
                        # Center item on index finger
                        new_x = idx_tip[0] - img_w // 2
                        new_y = idx_tip[1] - img_h // 2
                        item["pos"] = [new_x, new_y]
                        
                        # Visual feedback for grabbed item
                        cv2.putText(frame, f"Dragging: {item['name']}", (10, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # Released - drop the item
                        draggable_items[grabbed_idx]["selected"] = False
                        grabbed_idx = -1
                        
                else:
                    # Not holding anything - check for new grab
                    if distance < GRAB_DISTANCE_THRESHOLD:
                        # Check each item (reverse order for top-most first)
                        for idx in reversed(range(len(draggable_items))):
                            item = draggable_items[idx]
                            img_h, img_w = item["image"].shape[:2]
                            
                            if point_in_rect(idx_tip, item["pos"], (img_w, img_h)):
                                grabbed_idx = idx
                                item["selected"] = True
                                print(f"Grabbed: {item['name']}")
                                break

        # Render all draggable items
        for idx, item in enumerate(draggable_items):
            x, y = item["pos"]
            
            # Add visual feedback for selected items
            if item["selected"]:
                img_h, img_w = item["image"].shape[:2]
                cv2.rectangle(frame, (x-2, y-2), (x+img_w+2, y+img_h+2), (0, 255, 0), 2)
            
            frame = overlay_png(frame, item["image"], x, y)

        # Display instructions
        instructions = [
            "Pinch fingers together to grab items",
            "Move while pinching to drag",
            "Release to drop | ESC to exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Touchless Drag-and-Drop", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('r'):  # Reset positions
            start_x, start_y = 100, 100
            for idx, item in enumerate(draggable_items):
                item["pos"] = [start_x + idx * 200, start_y]
                item["selected"] = False
            grabbed_idx = -1
            print("Reset all item positions")

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # ------------------------ CLEANUP ------------------------
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")