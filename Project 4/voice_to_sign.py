import speech_recognition as sr
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from collections import deque
import tkinter as tk
from tkinter import ttk, scrolledtext
import json

class SignLanguageConverter:
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Text buffer for recognized speech
        self.text_buffer = deque(maxlen=10)
        self.current_text = ""
        
        # Sign language dictionary (basic ASL alphabet and common words)
        self.sign_dict = self.load_sign_dictionary()
        
        # GUI setup
        self.setup_gui()
        
        # Threading control
        self.listening = False
        self.camera_active = False
        
    def load_sign_dictionary(self):
        """Load basic sign language mappings"""
        return {
            'hello': 'Wave hand',
            'thank you': 'Touch chin, move hand forward',
            'please': 'Circular motion on chest',
            'yes': 'Nod fist up and down',
            'no': 'Index and middle finger tap thumb',
            'sorry': 'Circular motion on chest with fist',
            'help': 'Flat hand on opposite fist, lift both',
            'water': 'W handshape tap chin',
            'eat': 'Bring fingertips to mouth',
            'good': 'Flat hand from chin forward',
            # Add alphabet
            **{chr(i): f'Letter {chr(i).upper()} handshape' for i in range(ord('a'), ord('z')+1)}
        }
    
    def setup_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Voice to Sign Language Converter")
        self.root.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Voice to Sign Language Converter", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Speech recognition section
        speech_frame = ttk.LabelFrame(main_frame, text="Speech Recognition", padding="10")
        speech_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.listen_btn = ttk.Button(speech_frame, text="Start Listening", 
                                    command=self.toggle_listening)
        self.listen_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.status_label = ttk.Label(speech_frame, text="Ready to listen...")
        self.status_label.grid(row=0, column=1)
        
        # Recognized text display
        text_frame = ttk.LabelFrame(main_frame, text="Recognized Speech", padding="10")
        text_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.text_display = scrolledtext.ScrolledText(text_frame, height=5, width=70)
        self.text_display.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Sign language translation
        sign_frame = ttk.LabelFrame(main_frame, text="Sign Language Translation", padding="10")
        sign_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.sign_display = scrolledtext.ScrolledText(sign_frame, height=8, width=70)
        self.sign_display.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Camera controls
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Controls", padding="10")
        camera_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.camera_btn = ttk.Button(camera_frame, text="Start Camera", 
                                    command=self.toggle_camera)
        self.camera_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.camera_status = ttk.Label(camera_frame, text="Camera off")
        self.camera_status.grid(row=0, column=1)
        
        # Clear button
        clear_btn = ttk.Button(main_frame, text="Clear All", command=self.clear_all)
        clear_btn.grid(row=5, column=0, columnspan=2, pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
    def toggle_listening(self):
        """Start/stop speech recognition"""
        if not self.listening:
            self.listening = True
            self.listen_btn.config(text="Stop Listening")
            self.status_label.config(text="Listening...")
            
            # Start listening in a separate thread
            self.listen_thread = threading.Thread(target=self.listen_for_speech)
            self.listen_thread.daemon = True
            self.listen_thread.start()
        else:
            self.listening = False
            self.listen_btn.config(text="Start Listening")
            self.status_label.config(text="Stopped listening")
    
    def listen_for_speech(self):
        """Continuous speech recognition"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
        while self.listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                # Recognize speech
                text = self.recognizer.recognize_google(audio)
                
                # Update GUI in main thread
                self.root.after(0, self.update_text_display, text)
                
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                self.root.after(0, self.update_status, "Could not understand audio")
            except sr.RequestError as e:
                self.root.after(0, self.update_status, f"Error: {e}")
    
    def update_text_display(self, text):
        """Update the text display with recognized speech"""
        self.text_display.insert(tk.END, text + "\n")
        self.text_display.see(tk.END)
        
        # Convert to sign language
        self.convert_to_sign_language(text)
        
        self.status_label.config(text="Recognition successful")
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
    
    def convert_to_sign_language(self, text):
        """Convert text to sign language instructions"""
        words = text.lower().split()
        sign_instructions = []
        
        for word in words:
            if word in self.sign_dict:
                sign_instructions.append(f"{word.upper()}: {self.sign_dict[word]}")
            else:
                # Spell out unknown words letter by letter
                spelled_word = []
                for letter in word:
                    if letter.isalpha() and letter in self.sign_dict:
                        spelled_word.append(f"{letter.upper()}")
                if spelled_word:
                    sign_instructions.append(f"Spell '{word.upper()}': " + " → ".join(spelled_word))
        
        # Display sign language instructions
        if sign_instructions:
            instruction_text = "\n".join(sign_instructions) + "\n" + "="*50 + "\n"
            self.sign_display.insert(tk.END, instruction_text)
            self.sign_display.see(tk.END)
    
    def toggle_camera(self):
        """Start/stop camera for hand tracking"""
        if not self.camera_active:
            self.camera_active = True
            self.camera_btn.config(text="Stop Camera")
            self.camera_status.config(text="Camera active")
            
            # Start camera in separate thread
            self.camera_thread = threading.Thread(target=self.run_camera)
            self.camera_thread.daemon = True
            self.camera_thread.start()
        else:
            self.camera_active = False
            self.camera_btn.config(text="Start Camera")
            self.camera_status.config(text="Camera off")
    
    def run_camera(self):
        """Run camera for hand tracking"""
        cap = cv2.VideoCapture(0)
        
        while self.camera_active:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                              self.mp_hands.HAND_CONNECTIONS)
            
            # Add instructions to frame
            cv2.putText(frame, "Hand Tracking - Practice Sign Language", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to close camera", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Sign Language Practice', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Update GUI
        self.root.after(0, self.camera_stopped)
    
    def camera_stopped(self):
        """Update GUI when camera stops"""
        self.camera_active = False
        self.camera_btn.config(text="Start Camera")
        self.camera_status.config(text="Camera off")
    
    def clear_all(self):
        """Clear all text displays"""
        self.text_display.delete(1.0, tk.END)
        self.sign_display.delete(1.0, tk.END)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

# Additional utility functions
def install_requirements():
    """Function to help users install required packages"""
    requirements = [
        "speechrecognition",
        "pyaudio",
        "opencv-python",
        "mediapipe",
        "numpy"
    ]
    
    print("To install required packages, run:")
    for req in requirements:
        print(f"pip install {req}")

if __name__ == "__main__":
    print("Voice to Sign Language Converter")
    print("=================================")
    print("This application converts speech to sign language instructions.")
    print("Make sure you have installed the required packages.")
    print()
    
    try:
        # Create and run the converter
        converter = SignLanguageConverter()
        converter.run()
    except ImportError as e:
        print(f"Missing required package: {e}")
        print()
        install_requirements()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Make sure your microphone and camera are properly connected.")