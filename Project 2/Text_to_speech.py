import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
from datetime import datetime
import random
import re

class VoiceChatbot:
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.setup_tts()
        
        # Chat state
        self.is_listening = False
        self.conversation_history = []
        
        # Response queue for threading
        self.response_queue = queue.Queue()
        
        print("Voice Chatbot initialized!")
        print("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        print("Calibration complete!")
    
    def setup_tts(self):
        """Configure text-to-speech settings"""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Try to set a pleasant voice (usually female voice is at index 1)
            if len(voices) > 1:
                self.tts_engine.setProperty('voice', voices[1].id)
        
        # Set speech rate and volume
        self.tts_engine.setProperty('rate', 180)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
    
    def speak(self, text):
        """Convert text to speech"""
        print(f"Bot: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def listen(self):
        """Listen for voice input and convert to text"""
        try:
            with self.microphone as source:
                print("Listening... (speak now)")
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            print("Processing speech...")
            # Convert speech to text
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text.lower()
            
        except sr.WaitTimeoutError:
            return "timeout"
        except sr.UnknownValueError:
            return "unclear"
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return "error"
    
    def generate_response(self, user_input):
        """Generate chatbot responses based on user input"""
        user_input = user_input.lower().strip()
        
        # Exit commands
        if any(phrase in user_input for phrase in ["goodbye", "bye", "exit", "quit", "stop"]):
            return "Goodbye! It was nice talking with you."
        
        # Greeting responses
        if any(phrase in user_input for phrase in ["hello", "hi", "hey", "greetings"]):
            greetings = [
                "Hello! How can I help you today?",
                "Hi there! What would you like to talk about?",
                "Hey! I'm here and ready to chat.",
                "Greetings! How are you doing?"
            ]
            return random.choice(greetings)
        
        # Time-related queries
        if "time" in user_input:
            current_time = datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}."
        
        # Date-related queries
        if "date" in user_input or "today" in user_input:
            current_date = datetime.now().strftime("%B %d, %Y")
            return f"Today is {current_date}."
        
        # Weather (simulated response)
        if "weather" in user_input:
            weather_responses = [
                "I don't have access to real-time weather data, but I hope it's nice where you are!",
                "I wish I could check the weather for you. Try asking a weather app or looking outside!",
                "I can't access weather information right now, but I hope you're having a pleasant day!"
            ]
            return random.choice(weather_responses)
        
        # Name-related queries
        if "your name" in user_input or "who are you" in user_input:
            return "I'm your voice assistant chatbot! You can call me Bot. What's your name?"
        
        # How are you responses
        if "how are you" in user_input:
            responses = [
                "I'm doing great, thank you for asking! How are you?",
                "I'm functioning perfectly and ready to help! How about you?",
                "I'm excellent! Thanks for checking in. How's your day going?"
            ]
            return random.choice(responses)
        
        # Help requests
        if "help" in user_input:
            return "I can chat with you, tell you the time or date, and respond to various questions. Just speak naturally and I'll do my best to help!"
        
        # Jokes
        if "joke" in user_input or "funny" in user_input:
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "I told my wife she was drawing her eyebrows too high. She looked surprised!",
                "Why don't programmers like nature? It has too many bugs!",
                "What do you call a bear with no teeth? A gummy bear!"
            ]
            return random.choice(jokes)
        
        # Default responses for unrecognized input
        default_responses = [
            "That's interesting! Can you tell me more about that?",
            "I see. What else would you like to talk about?",
            "That's a good point. What do you think about it?",
            "I'm not sure I fully understand, but I'm listening. Can you elaborate?",
            "Hmm, that's something to think about. What's your opinion on that?",
            "I appreciate you sharing that with me. What else is on your mind?"
        ]
        return random.choice(default_responses)
    
    def start_conversation(self):
        """Main conversation loop"""
        self.speak("Hello! I'm your voice assistant. You can talk to me naturally. Say 'goodbye' when you want to stop.")
        
        while True:
            try:
                # Listen for user input
                user_input = self.listen()
                
                # Handle special cases
                if user_input == "timeout":
                    self.speak("I didn't hear anything. Are you still there?")
                    continue
                elif user_input == "unclear":
                    self.speak("I didn't catch that clearly. Could you please repeat?")
                    continue
                elif user_input == "error":
                    self.speak("I'm having trouble with speech recognition. Please try again.")
                    continue
                
                # Store conversation
                self.conversation_history.append(("User", user_input))
                
                # Generate and speak response
                response = self.generate_response(user_input)
                self.conversation_history.append(("Bot", response))
                
                self.speak(response)
                
                # Check for exit condition
                if any(phrase in user_input for phrase in ["goodbye", "bye", "exit", "quit", "stop"]):
                    break
                    
            except KeyboardInterrupt:
                self.speak("Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                self.speak("I encountered an error. Let me try to continue.")
    
    def show_conversation_history(self):
        """Display the conversation history"""
        print("\n=== Conversation History ===")
        for speaker, message in self.conversation_history:
            print(f"{speaker}: {message}")
        print("=" * 30)

def main():
    """Main function to run the voice chatbot"""
    print("Starting Voice Recognition Chatbot...")
    print("Make sure your microphone is working and you have an internet connection.")
    print("Press Ctrl+C to force quit if needed.\n")
    
    try:
        # Create and start the chatbot
        chatbot = VoiceChatbot()
        chatbot.start_conversation()
        
        # Show conversation history at the end
        chatbot.show_conversation_history()
        
    except Exception as e:
        print(f"Failed to start chatbot: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install SpeechRecognition pyttsx3 pyaudio")

if __name__ == "__main__":
    main()