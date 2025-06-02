import random
import time
import json
import os
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as patches

class VisualSignLanguageLearner:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = None
        
        self.user_progress = {
            'letters_learned': set(),
            'numbers_learned': set(),
            'words_learned': set(),
            'total_practice_time': 0,
            'correct_answers': 0,
            'total_attempts': 0,
            'level': 1,
            'camera_verifications': 0
        }
        
        # ASL Alphabet with visual coordinates and descriptions
        self.alphabet_signs = {
            'A': {
                'description': "Make a fist with thumb alongside fingers",
                'visual': self.draw_letter_A,
                'landmarks': 'closed_fist_thumb_side'
            },
            'B': {
                'description': "Hold four fingers up straight, thumb across palm",
                'visual': self.draw_letter_B,
                'landmarks': 'four_fingers_up'
            },
            'C': {
                'description': "Curve fingers and thumb to form a 'C' shape",
                'visual': self.draw_letter_C,
                'landmarks': 'curved_c_shape'
            },
            'D': {
                'description': "Point index finger up, other fingers and thumb form 'O'",
                'visual': self.draw_letter_D,
                'landmarks': 'index_up_others_circle'
            },
            'E': {
                'description': "Bend all fingertips to touch thumb",
                'visual': self.draw_letter_E,
                'landmarks': 'fingertips_to_thumb'
            },
            'L': {
                'description': "Make 'L' shape with thumb and index finger",
                'visual': self.draw_letter_L,
                'landmarks': 'l_shape'
            },
            'O': {
                'description': "Make 'O' shape with all fingers and thumb",
                'visual': self.draw_letter_O,
                'landmarks': 'circle_shape'
            },
            'V': {
                'description': "Make 'V' with index and middle finger",
                'visual': self.draw_letter_V,
                'landmarks': 'peace_sign'
            },
            'Y': {
                'description': "Hold up thumb and pinky",
                'visual': self.draw_letter_Y,
                'landmarks': 'thumb_pinky_up'
            }
        }
        
        # Numbers with visual representations
        self.number_signs = {
            '1': {
                'description': "Point index finger up",
                'visual': self.draw_number_1,
                'landmarks': 'index_finger_up'
            },
            '2': {
                'description': "Hold up index and middle finger in 'V'",
                'visual': self.draw_number_2,
                'landmarks': 'peace_sign'
            },
            '3': {
                'description': "Hold up thumb, index, and middle finger",
                'visual': self.draw_number_3,
                'landmarks': 'three_fingers_up'
            },
            '5': {
                'description': "Hold up all five fingers spread apart",
                'visual': self.draw_number_5,
                'landmarks': 'open_hand'
            }
        }
        
        # Common words
        self.common_words = {
            'HELLO': {
                'description': "Wave hand side to side",
                'visual': self.draw_word_hello,
                'landmarks': 'open_hand_wave'
            },
            'THANK YOU': {
                'description': "Touch fingertips to chin, move hand forward",
                'visual': self.draw_word_thankyou,
                'landmarks': 'fingertips_to_chin'
            },
            'YES': {
                'description': "Make fist, nod wrist up and down",
                'visual': self.draw_word_yes,
                'landmarks': 'fist_nod'
            }
        }
        
        self.load_progress()
    
    def draw_hand_base(self, ax, title):
        """Draw base hand outline"""
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Draw palm
        palm = FancyBboxPatch((3, 2), 4, 6, boxstyle="round,pad=0.3", 
                             facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(palm)
        
        return ax
    
    def draw_letter_A(self, ax):
        """Draw letter A visualization"""
        ax = self.draw_hand_base(ax, "Letter A - Fist with Thumb")
        
        # Closed fist
        fist = Circle((5, 5), 2, facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(fist)
        
        # Thumb alongside
        thumb = patches.Ellipse((3.5, 5), 0.8, 1.5, facecolor='#FFE4B5', 
                               edgecolor='black', linewidth=2)
        ax.add_patch(thumb)
        
        ax.text(5, 1, "Make a fist with thumb\nalongside fingers", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_letter_B(self, ax):
        """Draw letter B visualization"""
        ax = self.draw_hand_base(ax, "Letter B - Four Fingers Up")
        
        # Four fingers up
        fingers = [(4, 8), (5, 9), (6, 8.5), (7, 8)]
        for i, (x, y) in enumerate(fingers):
            finger = patches.Rectangle((x-0.3, 5), 0.6, y-5, 
                                     facecolor='#FFE4B5', edgecolor='black', linewidth=2)
            ax.add_patch(finger)
        
        # Thumb across palm
        thumb = patches.Rectangle((3, 4), 1.5, 0.6, 
                                facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(thumb)
        
        ax.text(5, 1, "Hold four fingers up straight,\nthumb across palm", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_letter_C(self, ax):
        """Draw letter C visualization"""
        ax = self.draw_hand_base(ax, "Letter C - Curved Shape")
        
        # C-shaped curve
        c_shape = patches.Arc((5, 5), 4, 6, angle=0, theta1=30, theta2=330, 
                             edgecolor='black', linewidth=4)
        ax.add_patch(c_shape)
        
        ax.text(5, 1, "Curve fingers and thumb\nto form a 'C' shape", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_letter_D(self, ax):
        """Draw letter D visualization"""
        ax = self.draw_hand_base(ax, "Letter D - Index Up, Others Circle")
        
        # Index finger up
        index = patches.Rectangle((4.7, 5), 0.6, 4, 
                                facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(index)
        
        # Other fingers form circle
        circle = Circle((6, 4), 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        ax.text(5, 1, "Point index finger up,\nother fingers form 'O'", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_letter_E(self, ax):
        """Draw letter E visualization"""
        ax = self.draw_hand_base(ax, "Letter E - Fingertips to Thumb")
        
        # Curved fingers touching thumb
        for i, x in enumerate([4, 5, 6, 7]):
            finger = patches.Arc((x, 5), 1, 2, angle=0, theta1=0, theta2=180, 
                               edgecolor='black', linewidth=2)
            ax.add_patch(finger)
        
        ax.text(5, 1, "Bend all fingertips\nto touch thumb", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_letter_L(self, ax):
        """Draw letter L visualization"""
        ax = self.draw_hand_base(ax, "Letter L - L Shape")
        
        # Index finger up
        index = patches.Rectangle((4.7, 5), 0.6, 4, 
                                facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(index)
        
        # Thumb out
        thumb = patches.Rectangle((3, 4.7), 2, 0.6, 
                                facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(thumb)
        
        ax.text(5, 1, "Make 'L' shape with\nthumb and index finger", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_letter_O(self, ax):
        """Draw letter O visualization"""
        ax = self.draw_hand_base(ax, "Letter O - Circle Shape")
        
        # O shape with all fingers
        circle = Circle((5, 5), 2, fill=False, edgecolor='black', linewidth=4)
        ax.add_patch(circle)
        
        ax.text(5, 1, "Make 'O' shape with\nall fingers and thumb", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_letter_V(self, ax):
        """Draw letter V visualization"""
        ax = self.draw_hand_base(ax, "Letter V - Peace Sign")
        
        # Two fingers up in V
        finger1 = patches.Rectangle((4.2, 5), 0.6, 4, angle=15,
                                  facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        finger2 = patches.Rectangle((5.2, 5), 0.6, 4, angle=-15,
                                  facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(finger1)
        ax.add_patch(finger2)
        
        ax.text(5, 1, "Make 'V' with index\nand middle finger", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_letter_Y(self, ax):
        """Draw letter Y visualization"""
        ax = self.draw_hand_base(ax, "Letter Y - Thumb and Pinky")
        
        # Thumb up
        thumb = patches.Rectangle((3, 5), 0.6, 3, 
                                facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(thumb)
        
        # Pinky up
        pinky = patches.Rectangle((7, 5), 0.6, 3, 
                               facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(pinky)
        
        ax.text(5, 1, "Hold up thumb and pinky\n('Hang loose' sign)", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_number_1(self, ax):
        """Draw number 1 visualization"""
        ax = self.draw_hand_base(ax, "Number 1 - Index Finger")
        
        # Index finger up
        index = patches.Rectangle((4.7, 5), 0.6, 4, 
                                facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(index)
        
        ax.text(5, 1, "Point index finger up", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_number_2(self, ax):
        """Draw number 2 visualization"""
        return self.draw_letter_V(ax)  # Same as letter V
    
    def draw_number_3(self, ax):
        """Draw number 3 visualization"""
        ax = self.draw_hand_base(ax, "Number 3 - Three Fingers")
        
        # Three fingers up
        fingers = [(4, 8.5), (5, 9), (6, 8.5)]
        for x, y in fingers:
            finger = patches.Rectangle((x-0.3, 5), 0.6, y-5, 
                                     facecolor='#FFE4B5', edgecolor='black', linewidth=2)
            ax.add_patch(finger)
        
        ax.text(5, 1, "Hold up thumb, index,\nand middle finger", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_number_5(self, ax):
        """Draw number 5 visualization"""
        ax = self.draw_hand_base(ax, "Number 5 - Open Hand")
        
        # Five fingers spread
        fingers = [(3.5, 7), (4.5, 8.5), (5.5, 9), (6.5, 8.5), (7.5, 7)]
        for i, (x, y) in enumerate(fingers):
            angle = (i - 2) * 15  # Spread fingers
            finger = patches.Rectangle((x-0.3, 5), 0.6, y-5, angle=angle,
                                     facecolor='#FFE4B5', edgecolor='black', linewidth=2)
            ax.add_patch(finger)
        
        ax.text(5, 1, "Hold up all five fingers\nspread apart", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_word_hello(self, ax):
        """Draw HELLO gesture"""
        ax = self.draw_hand_base(ax, "HELLO - Wave Hand")
        
        # Open hand with motion lines
        for x in [3.5, 4.5, 5.5, 6.5, 7.5]:
            finger = patches.Rectangle((x-0.2, 5), 0.4, 3, 
                                     facecolor='#FFE4B5', edgecolor='black', linewidth=2)
            ax.add_patch(finger)
        
        # Motion lines
        for i in range(3):
            ax.plot([2+i*0.5, 8+i*0.5], [7, 7], 'r--', alpha=0.7, linewidth=2)
        
        ax.text(5, 1, "Wave hand side to side", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_word_thankyou(self, ax):
        """Draw THANK YOU gesture"""
        ax = self.draw_hand_base(ax, "THANK YOU - Fingertips to Chin")
        
        # Hand near chin area
        for x in [4, 5, 6, 7]:
            finger = patches.Rectangle((x-0.2, 8), 0.4, 2, 
                                     facecolor='#FFE4B5', edgecolor='black', linewidth=2)
            ax.add_patch(finger)
        
        # Arrow showing forward movement
        ax.arrow(5, 9, 2, 0, head_width=0.3, head_length=0.3, 
                fc='red', ec='red', linewidth=2)
        
        ax.text(5, 1, "Touch fingertips to chin,\nmove hand forward", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def draw_word_yes(self, ax):
        """Draw YES gesture"""
        ax = self.draw_hand_base(ax, "YES - Fist Nod")
        
        # Fist
        fist = Circle((5, 5), 1.5, facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(fist)
        
        # Up and down arrows
        ax.arrow(5, 7, 0, 1, head_width=0.3, head_length=0.3, 
                fc='blue', ec='blue', linewidth=2)
        ax.arrow(5, 8.5, 0, -1, head_width=0.3, head_length=0.3, 
                fc='blue', ec='blue', linewidth=2)
        
        ax.text(5, 1, "Make fist, nod wrist\nup and down", 
                ha='center', fontsize=12, weight='bold')
        
        return ax
    
    def show_visual_sign(self, sign_data, sign_name):
        """Display visual representation of a sign"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        # Draw the sign
        sign_data['visual'](ax)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to ensure window opens
        
        return fig
    
    def initialize_camera(self):
        """Initialize camera for hand detection"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Cannot open camera. Visual verification disabled.")
                return False
        return True
    
    def detect_hand_landmarks(self, image):
        """Detect hand landmarks using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]  # Return first hand
        return None
    
    def verify_sign_with_camera(self, expected_sign, sign_name):
        """Use camera to verify if user is making the correct sign"""
        if not self.initialize_camera():
            return self.fallback_verification(sign_name)
        
        print(f"\n📷 Camera Verification for '{sign_name}'")
        print("Position your hand in front of the camera...")
        print("Press 'q' to quit camera mode")
        print("Press 'c' to capture and verify your sign")
        
        verification_result = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand landmarks
            landmarks = self.detect_hand_landmarks(frame)
            
            if landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Add instruction text
                cv2.putText(frame, f"Making sign: {sign_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'c' to verify", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Show your hand to camera", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Sign Language Verification', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and landmarks:
                # Verify the sign
                verification_result = self.analyze_hand_landmarks(landmarks, expected_sign)
                if verification_result:
                    print("✅ Great! Your sign looks correct!")
                    cv2.putText(frame, "CORRECT SIGN!", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow('Sign Language Verification', frame)
                    cv2.waitKey(2000)  # Show success for 2 seconds
                else:
                    print("❌ Try again - check the visual guide")
                    cv2.putText(frame, "TRY AGAIN", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow('Sign Language Verification', frame)
                    cv2.waitKey(2000)  # Show message for 2 seconds
                break
        
        cv2.destroyAllWindows()
        return verification_result
    
    def analyze_hand_landmarks(self, landmarks, expected_sign):
        """Analyze hand landmarks to verify sign (simplified version)"""
        # This is a simplified verification - in a real app, you'd need
        # sophisticated gesture recognition algorithms
        
        # Get landmark positions
        landmark_points = []
        for lm in landmarks.landmark:
            landmark_points.append([lm.x, lm.y])
        
        # Simple heuristic checks based on finger positions
        if expected_sign == 'peace_sign':  # V or 2
            return self.check_peace_sign(landmark_points)
        elif expected_sign == 'open_hand':  # 5 or Hello
            return self.check_open_hand(landmark_points)
        elif expected_sign == 'index_finger_up':  # 1
            return self.check_index_up(landmark_points)
        elif expected_sign == 'l_shape':  # L
            return self.check_l_shape(landmark_points)
        elif expected_sign == 'thumb_pinky_up':  # Y
            return self.check_y_shape(landmark_points)
        else:
            # For other signs, use basic hand detection
            return len(landmark_points) == 21  # Just check if hand is detected
    
    def check_peace_sign(self, points):
        """Check if hand is making peace sign (V)"""
        # Simplified: check if index and middle finger are extended
        # Points 8 and 12 are fingertips, 6 and 10 are middle joints
        index_tip_y = points[8][1]
        middle_tip_y = points[12][1]
        ring_tip_y = points[16][1]
        pinky_tip_y = points[20][1]
        
        # Index and middle should be higher (lower y) than ring and pinky
        return (index_tip_y < ring_tip_y) and (middle_tip_y < ring_tip_y)
    
    def check_open_hand(self, points):
        """Check if hand is open (all fingers extended)"""
        fingertips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        finger_mcp = [5, 9, 13, 17]   # Knuckles
        
        extended_count = 0
        for tip, mcp in zip(fingertips, finger_mcp):
            if points[tip][1] < points[mcp][1]:  # Tip higher than knuckle
                extended_count += 1
        
        return extended_count >= 3  # At least 3 fingers extended
    
    def check_index_up(self, points):
        """Check if only index finger is up"""
        index_tip_y = points[8][1]
        middle_tip_y = points[12][1]
        ring_tip_y = points[16][1]
        
        # Index should be significantly higher than others
        return (index_tip_y < middle_tip_y - 0.05) and (index_tip_y < ring_tip_y - 0.05)
    
    def check_l_shape(self, points):
        """Check if hand is making L shape"""
        thumb_tip = points[4]
        index_tip = points[8]
        
        # Check if thumb and index form roughly perpendicular lines
        # Simplified check: thumb should be to the left, index should be up
        return (thumb_tip[0] < index_tip[0]) and (index_tip[1] < thumb_tip[1])
    
    def check_y_shape(self, points):
        """Check if hand is making Y shape (thumb and pinky up)"""
        thumb_tip_y = points[4][1]
        pinky_tip_y = points[20][1]
        middle_tip_y = points[12][1]
        
        # Thumb and pinky should be higher than middle finger
        return (thumb_tip_y < middle_tip_y) and (pinky_tip_y < middle_tip_y)
    
    def fallback_verification(self, sign_name):
        """Fallback verification when camera is not available"""
        print(f"\n📝 Manual Verification for '{sign_name}'")
        print("Camera not available. Please verify manually:")
        
        response = input("Are you making the correct sign? (y/n): ").lower().strip()
        return response == 'y'
    
    def save_progress(self):
        """Save user progress to file"""
        progress_data = {
            'letters_learned': list(self.user_progress['letters_learned']),
            'numbers_learned': list(self.user_progress['numbers_learned']),
            'words_learned': list(self.user_progress['words_learned']),
            'total_practice_time': self.user_progress['total_practice_time'],
            'correct_answers': self.user_progress['correct_answers'],
            'total_attempts': self.user_progress['total_attempts'],
            'level': self.user_progress['level'],
            'camera_verifications': self.user_progress['camera_verifications']
        }
        
        try:
            with open('visual_sign_progress.json', 'w') as f:
                json.dump(progress_data, f)
        except:
            pass
    
    def load_progress(self):
        """Load user progress from file"""
        try:
            if os.path.exists('visual_sign_progress.json'):
                with open('visual_sign_progress.json', 'r') as f:
                    progress_data = json.load(f)
                    self.user_progress['letters_learned'] = set(progress_data.get('letters_learned', []))
                    self.user_progress['numbers_learned'] = set(progress_data.get('numbers_learned', []))
                    self.user_progress['words_learned'] = set(progress_data.get('words_learned', []))
                    self.user_progress['total_practice_time'] = progress_data.get('total_practice_time', 0)
                    self.user_progress['correct_answers'] = progress_data.get('correct_answers', 0)
                    self.user_progress['total_attempts'] = progress_data.get('total_attempts', 0)
                    self.user_progress['level'] = progress_data.get('level', 1)
                    self.user_progress['camera_verifications'] = progress_data.get('camera_verifications', 0)
        except:
            pass
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("    🤟 VISUAL SIGN LANGUAGE LEARNING APP 🤟")
        print("         With Camera Verification!")
        print("="*60)
        print("1. 📚 Learn Alphabet (A-Z) - Visual + Camera")
        print("2. 🔢 Learn Numbers (0-10) - Visual + Camera")
        print("3. 💬 Learn Common Words - Visual + Camera")
        print("4. 🎯 Practice Quiz with Visual Verification")
        print("5. 📊 View Progress")
        print("6. 🎮 Interactive Games")
        print("7. 📷 Test Camera Setup")
        print("8. ❌ Exit")
        print("="*60)
    
    def learn_alphabet_visual(self):
        """Learn alphabet with visual representation and camera verification"""
        print("\n🔤 VISUAL ASL ALPHABET LEARNING")
        print("-" * 40)
        
        available_letters = list(self.alphabet_signs.keys())
        
        while True:
            print(f"\nAvailable letters: {', '.join(available_letters)}")
            choice = input("\nEnter a letter to learn (or 'back' to return): ").upper().strip()
            
            if choice == 'BACK':
                break
            elif choice in self.alphabet_signs:
                sign_data = self.alphabet_signs[choice]
                
                print(f"\n✋ Learning Letter '{choice}':")
                print(f"Description: {sign_data['description']}")
                
                # Show visual representation
                print("\n🖼️  Showing visual guide...")
                fig = self.show_visual_sign(sign_data, f"Letter {choice}")
                
                input("\nStudy the visual guide, then press Enter to try...")
                plt.close(fig)
                
                # Camera verification
                print(f"\n📷 Now try making the sign for '{choice}'!")
                verification_success = self.verify_sign_with_camera(
                    sign_data['landmarks'], f"Letter {choice}"
                )
                
                if verification_success:
                    print("🎉 Excellent! You got it right!")
                    self.user_progress['letters_learned'].add(choice)
                    self.user_progress['correct_answers'] += 1
                    self.user_progress['camera_verifications'] += 1
                else:
                    print("💪 Keep practicing! Try again when you're ready.")
                
                self.user_progress['total_attempts'] += 1
            else:
                print("Invalid letter. Please try again.")
    
    def learn_numbers_visual(self):
        """Learn numbers with visual representation and camera verification"""
        print("\n🔢 VISUAL ASL NUMBERS LEARNING")
        print("-" * 40)
        
        available_numbers = list(self.number_signs.keys())
        
        while True:
            print(f"\nAvailable numbers: {', '.join(available_numbers)}")
            choice = input("\nEnter a number to learn (or 'back' to return): ").strip()
            
            if choice.upper() == 'BACK':
                break
            elif choice in self.number_signs:
                sign_data = self.number_signs[choice]
                
                print(f"\n✋ Learning Number '{choice}':")
                print(f"Description: {sign_data['description']}")
                
                # Show visual representation
                print("\n🖼️  Showing visual guide...")
                fig = self.show_visual_sign(sign_data, f"Number {choice}")
                
                input("\nStudy the visual guide, then press Enter to try...")
                plt.close(fig)
                
                # Camera verification
                print(f"\n📷 Now try making the sign for '{choice}'!")
                verification_success = self.verify_sign_with_camera(
                    sign_data['landmarks'], f"Number {choice}"
                )
                
                if verification_success:
                    print("🎉 Perfect! You nailed it!")
                    self.user_progress['numbers_learned'].add(choice)
                    self.user_progress['correct_answers'] += 1
                    self.user_progress['camera_verifications'] += 1
                else:
                    print("💪 Keep practicing! The visual guide is there to help.")
                
                self.user_progress['total_attempts'] += 1
            else:
                print("Invalid number. Please try again.")
    
    def learn_words_visual(self):
        """Learn words with visual representation and camera verification"""
        print("\n💬 VISUAL ASL COMMON WORDS")
        print("-" * 40)
        
        available_words = list(self.common_words.keys())
        
        while True:
            print(f"\nAvailable words:")
            for i, word in enumerate(available_words, 1):
                status = "✅" if word in self.user_progress['words_learned'] else "⭕"
                print(f"  {i}. {word} {status}")
            
            choice = input(f"\nEnter word number (1-{len(available_words)}) or 'back': ").strip()
            
            if choice.upper() == 'BACK':
                break
            
            try:
                word_index = int(choice) - 1
                if 0 <= word_index < len(available_words):
                    word = available_words[word_index]
                    sign_data = self.common_words[word]
                    
                    print(f"\n✋ Learning Word '{word}':")
                    print(f"Description: {sign_data['description']}")
                    
                    # Show visual representation
                    print("\n🖼️  Showing visual guide...")
                    fig = self.show_visual_sign(sign_data, f"Word: {word}")
                    
                    input("\nStudy the visual guide, then press Enter to try...")
                    plt.close(fig)
                    
                    # Camera verification
                    print(f"\n📷 Now try making the sign for '{word}'!")
                    verification_success = self.verify_sign_with_camera(
                        sign_data['landmarks'], f"Word: {word}"
                    )
                    
                    if verification_success:
                        print("🎉 Amazing! You've got it!")
                        self.user_progress['words_learned'].add(word)
                        self.user_progress['correct_answers'] += 1
                        self.user_progress['camera_verifications'] += 1
                    else:
                        print("💪 Practice makes perfect! Try again when ready.")
                    
                    self.user_progress['total_attempts'] += 1
                else:
                    print("Invalid number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    def visual_practice_quiz(self):
        """Interactive quiz with visual guides and camera verification"""
        print("\n🎯 VISUAL PRACTICE QUIZ")
        print("-" * 40)
        
        # Combine all learned items
        quiz_items = {}
        
        for letter in self.user_progress['letters_learned']:
            if letter in self.alphabet_signs:
                quiz_items[f"Letter {letter}"] = self.alphabet_signs[letter]
        
        for number in self.user_progress['numbers_learned']:
            if number in self.number_signs:
                quiz_items[f"Number {number}"] = self.number_signs[number]
        
        for word in self.user_progress['words_learned']:
            if word in self.common_words:
                quiz_items[f"Word {word}"] = self.common_words[word]
        
        if not quiz_items:
            print("You haven't learned any signs yet! Go learn some first.")
            return
        
        score = 0
        total_questions = min(5, len(quiz_items))
        questions = random.sample(list(quiz_items.items()), total_questions)
        
        start_time = time.time()
        
        print(f"Ready for {total_questions} visual verification questions? Let's go!\n")
        
        for i, (item_name, sign_data) in enumerate(questions, 1):
            print(f"\nQuestion {i}/{total_questions}:")
            print(f"Make the sign for: {item_name}")
            
            # Show visual guide briefly
            show_guide = input("Do you want to see the visual guide first? (y/n): ").lower().strip()
            if show_guide == 'y':
                fig = self.show_visual_sign(sign_data, item_name)
                input("Study the guide, then press Enter to continue...")
                plt.close(fig)
            
            # Camera verification
            print(f"\n📷 Now make the sign for {item_name}!")
            verification_success = self.verify_sign_with_camera(
                sign_data['landmarks'], item_name
            )
            
            if verification_success:
                print("✅ Correct! Great job!")
                score += 1
                self.user_progress['correct_answers'] += 1
                self.user_progress['camera_verifications'] += 1
            else:
                print("❌ Not quite right, but good effort!")
            
            self.user_progress['total_attempts'] += 1
        
        end_time = time.time()
        practice_time = int(end_time - start_time)
        self.user_progress['total_practice_time'] += practice_time
        
        # Display results
        percentage = (score / total_questions) * 100
        print("\n" + "="*50)
        print("📊 VISUAL QUIZ RESULTS")
        print("="*50)
        print(f"Score: {score}/{total_questions} ({percentage:.1f}%)")
        print(f"Time taken: {practice_time} seconds")
        print(f"Camera verifications: {score}")
        
        if percentage >= 80:
            print("🎉 Outstanding! Your signs are looking great!")
            self.user_progress['level'] = min(5, self.user_progress['level'] + 1)
        elif percentage >= 60:
            print("👍 Good progress! Keep practicing with the camera!")
        else:
            print("💪 Keep going! The visual guides will help you improve!")
    
    def test_camera_setup(self):
        """Test camera setup and hand detection"""
        print("\n📷 CAMERA SETUP TEST")
        print("-" * 30)
        
        if not self.initialize_camera():
            print("❌ Camera initialization failed!")
            print("Make sure your camera is connected and not being used by other apps.")
            return
        
        print("✅ Camera initialized successfully!")
        print("Testing hand detection...")
        print("Show your hand to the camera. Press 'q' to quit test.")
        
        test_successful = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Cannot read from camera!")
                break
            
            frame = cv2.flip(frame, 1)
            landmarks = self.detect_hand_landmarks(frame)
            
            if landmarks:
                self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, "Hand detected! ✓", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                test_successful = True
            else:
                cv2.putText(frame, "Show your hand", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit test", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        if test_successful:
            print("✅ Camera test successful! Hand detection is working.")
        else:
            print("⚠️  Camera works but hand detection may need better lighting.")
        
        input("Press Enter to continue...")
    
    def view_visual_progress(self):
        """Display enhanced progress with camera verification stats"""
        print("\n📊 YOUR VISUAL LEARNING PROGRESS")
        print("="*50)
        
        total_letters = len(self.alphabet_signs)
        total_numbers = len(self.number_signs)
        total_words = len(self.common_words)
        
        letters_learned = len(self.user_progress['letters_learned'])
        numbers_learned = len(self.user_progress['numbers_learned'])
        words_learned = len(self.user_progress['words_learned'])
        
        print(f"📚 Letters learned: {letters_learned}/{total_letters} ({letters_learned/total_letters*100:.1f}%)")
        print(f"🔢 Numbers learned: {numbers_learned}/{total_numbers} ({numbers_learned/total_numbers*100:.1f}%)")
        print(f"💬 Words learned: {words_learned}/{total_words} ({words_learned/total_words*100:.1f}%)")
        print(f"📷 Camera verifications: {self.user_progress['camera_verifications']}")
        print(f"⏱️  Total practice time: {self.user_progress['total_practice_time']} seconds")
        print(f"✅ Correct answers: {self.user_progress['correct_answers']}")
        print(f"📝 Total attempts: {self.user_progress['total_attempts']}")
        print(f"🏆 Current level: {self.user_progress['level']}")
        
        if self.user_progress['total_attempts'] > 0:
            accuracy = (self.user_progress['correct_answers'] / self.user_progress['total_attempts']) * 100
            print(f"🎯 Overall accuracy: {accuracy:.1f}%")
        
        if self.user_progress['camera_verifications'] > 0:
            camera_accuracy = (self.user_progress['camera_verifications'] / self.user_progress['total_attempts']) * 100
            print(f"📷 Camera verification rate: {camera_accuracy:.1f}%")
        
        # Progress bar
        total_items = total_letters + total_numbers + total_words
        items_learned = letters_learned + numbers_learned + words_learned
        progress_percentage = (items_learned / total_items) * 100
        
        print(f"\n📈 Overall Progress: {items_learned}/{total_items} ({progress_percentage:.1f}%)")
        
        # Visual progress bar
        bar_length = 30
        filled_length = int(bar_length * progress_percentage // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"[{bar}] {progress_percentage:.1f}%")
        
        # Achievements
        print(f"\n🏆 ACHIEVEMENTS:")
        if letters_learned >= 5:
            print("   🔤 Alphabet Explorer - Learned 5+ letters!")
        if numbers_learned >= 3:
            print("   🔢 Number Master - Learned 3+ numbers!")
        if words_learned >= 2:
            print("   💬 Communicator - Learned 2+ words!")
        if self.user_progress['camera_verifications'] >= 10:
            print("   📷 Camera Pro - 10+ camera verifications!")
        if progress_percentage >= 50:
            print("   🌟 Halfway Hero - 50% completion!")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main application loop"""
        print("Welcome to the Visual Sign Language Learning App! 🤟")
        print("This app includes visual guides and camera verification!")
        
        # Check camera availability at startup
        if self.initialize_camera():
            print("✅ Camera detected and ready for verification!")
        else:
            print("⚠️  Camera not available - visual guides still work!")
        
        while True:
            self.display_menu()
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                self.learn_alphabet_visual()
            elif choice == '2':
                self.learn_numbers_visual()
            elif choice == '3':
                self.learn_words_visual()
            elif choice == '4':
                self.visual_practice_quiz()
            elif choice == '5':
                self.view_visual_progress()
            elif choice == '6':
                self.interactive_games_visual()
            elif choice == '7':
                self.test_camera_setup()
            elif choice == '8':
                self.save_progress()
                self.cleanup()
                print("\n👋 Thanks for learning sign language visually! Keep practicing!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
            
            # Auto-save progress
            self.save_progress()
    
    def interactive_games_visual(self):
        """Enhanced games with visual elements"""
        print("\n🎮 VISUAL INTERACTIVE GAMES")
        print("-" * 40)
        print("1. 🔤 Visual Alphabet Challenge")
        print("2. 🔢 Number Recognition Game")
        print("3. 💬 Word Gesture Game")
        print("4. 📷 Camera Speed Test")
        print("5. 🔄 Back to main menu")
        
        choice = input("\nChoose a game (1-5): ").strip()
        
        if choice == '1':
            self.visual_alphabet_challenge()
        elif choice == '2':
            self.visual_number_game()
        elif choice == '3':
            self.visual_word_game()
        elif choice == '4':
            self.camera_speed_test()
        elif choice == '5':
            return
        else:
            print("Invalid choice!")
    
    def visual_alphabet_challenge(self):
        """Alphabet challenge with visual verification"""
        print("\n🔤 VISUAL ALPHABET CHALLENGE")
        print("Make the signs as quickly as possible!")
        
        if not self.user_progress['letters_learned']:
            print("Learn some letters first!")
            return
        
        letters = random.sample(list(self.user_progress['letters_learned']), 
                               min(3, len(self.user_progress['letters_learned'])))
        start_time = time.time()
        correct_count = 0
        
        for i, letter in enumerate(letters, 1):
            print(f"\nRound {i}/3: Make the sign for letter '{letter}'")
            
            sign_data = self.alphabet_signs[letter]
            fig = self.show_visual_sign(sign_data, f"Letter {letter}")
            
            input("Ready? Press Enter then make the sign...")
            plt.close(fig)
            
            success = self.verify_sign_with_camera(sign_data['landmarks'], f"Letter {letter}")
            if success:
                correct_count += 1
                print("✅ Correct!")
            else:
                print("❌ Keep practicing!")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🏆 CHALLENGE RESULTS:")
        print(f"Score: {correct_count}/{len(letters)}")
        print(f"Time: {total_time:.2f} seconds")
        print(f"Average per sign: {total_time/len(letters):.2f} seconds")
        
        if correct_count == len(letters) and total_time < 60:
            print("🚀 Perfect speed run! You're a sign language champion!")
        elif correct_count >= len(letters) * 0.7:
            print("👍 Great job! Your accuracy is improving!")
        else:
            print("💪 Keep practicing! Speed and accuracy will come!")
    
    def camera_speed_test(self):
        """Test how quickly user can make different signs"""
        print("\n📷 CAMERA SPEED TEST")
        print("Make each sign as fast as you can when prompted!")
        
        if not self.initialize_camera():
            print("Camera required for this game!")
            return
        
        # Mix of learned signs
        all_learned = []
        for letter in self.user_progress['letters_learned']:
            all_learned.append(('Letter', letter, self.alphabet_signs[letter]))
        for number in self.user_progress['numbers_learned']:
            all_learned.append(('Number', number, self.number_signs[number]))
        
        if len(all_learned) < 3:
            print("Learn at least 3 signs to play this game!")
            return
        
        test_signs = random.sample(all_learned, min(5, len(all_learned)))
        total_time = 0
        successful_signs = 0
        
        print("Get ready! Signs will be announced one by one...")
        input("Press Enter to start the speed test...")
        
        for i, (sign_type, sign_name, sign_data) in enumerate(test_signs, 1):
            print(f"\n⚡ Sign {i}/5: Make {sign_type} '{sign_name}' NOW!")
            
            start = time.time()
            success = self.verify_sign_with_camera(sign_data['landmarks'], 
                                                 f"{sign_type} {sign_name}")
            end = time.time()
            
            sign_time = end - start
            total_time += sign_time
            
            if success:
                successful_signs += 1
                print(f"✅ Correct in {sign_time:.2f} seconds!")
            else:
                print(f"❌ Missed this one ({sign_time:.2f} seconds)")
        
        print(f"\n🏆 SPEED TEST RESULTS:")
        print(f"Successful signs: {successful_signs}/{len(test_signs)}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average per sign: {total_time/len(test_signs):.2f} seconds")
        
        if successful_signs == len(test_signs) and total_time < 30:
            print("⚡ Lightning speed! You're incredibly fast!")
        elif successful_signs >= len(test_signs) * 0.8:
            print("🚀 Excellent speed and accuracy!")
        else:
            print("💪 Good effort! Practice will make you faster!")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        plt.close('all')

# Installation and requirements note
print("""
📋 REQUIRED LIBRARIES:
Before running this app, install required packages:

pip install opencv-python mediapipe matplotlib numpy

🎯 FEATURES:
- Visual sign representations with matplotlib
- Real-time camera verification using MediaPipe
- Hand landmark detection and gesture recognition
- Progress tracking with camera verification stats
- Interactive games with visual feedback

📷 CAMERA REQUIREMENTS:
- Working webcam/camera
- Good lighting for hand detection
- Clear background for best results
""")

# Run the application
if __name__ == "__main__":
    try:
        app = VisualSignLanguageLearner()
        app.run()
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using the Visual Sign Language App!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure all required libraries are installed!")
        print("pip install opencv-python mediapipe matplotlib numpy")