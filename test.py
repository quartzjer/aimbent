import os
import json
import threading
import websocket
import pyaudio
import base64
import sys
import signal
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Constants
WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# Audio Configuration
AUDIO_FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1
RATE = 24000  # 24kHz
CHUNK = 1024  # Number of frames per buffer

# Global flag for interruption
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("\nExiting...")
    sys.exit(0)

# Register the signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(format=AUDIO_FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      output=True,
                                      frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            sys.exit(1)
        self.lock = threading.Lock()

    def play_audio(self, audio_bytes):
        with self.lock:
            try:
                self.stream.write(audio_bytes)
            except Exception as e:
                print(f"Error playing audio: {e}")

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        except Exception as e:
            print(f"Error closing audio stream: {e}")

class AudioSender:
    def __init__(self, ws):
        self.ws = ws
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(format=AUDIO_FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"Failed to open microphone stream: {e}")
            sys.exit(1)
        self.thread = threading.Thread(target=self.send_audio, daemon=True)
        self.running = False

    def start(self):
        self.running = True
        self.thread.start()
        print("Started sending audio from the microphone.")

    def send_audio(self):
        try:
            while self.running and not interrupted:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                audio_base64 = base64.b64encode(data).decode('utf-8')
                audio_event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64  # Corrected key name
                }
                self.ws.send(json.dumps(audio_event))
        except Exception as e:
            print(f"Error capturing/sending audio: {e}")
            self.running = False

    def commit_buffer(self):
        commit_event = {
            "type": "input_audio_buffer.commit"
        }
        try:
            self.ws.send(json.dumps(commit_event))
            print("Committed audio buffer.")
        except Exception as e:
            print(f"Failed to commit audio buffer: {e}")

    def stop(self):
        self.running = False
        self.commit_buffer()
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            print("Stopped sending audio.")
        except Exception as e:
            print(f"Error closing microphone stream: {e}")

class ChatStreaming:
    def __init__(self, api_key, verbose=False):
        self.api_key = api_key
        self.ws = None
        self.audio_player = AudioPlayer()
        self.audio_sender = None
        self.verbose = verbose
        self.audio_buffer = BytesIO()

    def log(self, message):
        if self.verbose:
            print(f"[DEBUG] {message}")

    def on_open(self, ws):
        print("Connected to server.")
        # Initialize and start audio sending
        self.audio_sender = AudioSender(ws)
        self.audio_sender.start()

        # Send session.update to initialize the session
        session_update_message = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": "You will be listening to a conversation, your job is to notice when you can be helpful and offer suggestions. You are not part of the conversation, you are only an observer.",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.8,
                },
                "temperature": 0.7,
                "max_response_output_tokens": 500
            }
        }
        ws.send(json.dumps(session_update_message))
        print("Session initialized.")

    def on_message(self, ws, message):
        try:
            event = json.loads(message)
            event_type = event.get("type")

            if event_type == "error":
                error = event.get("error", {})
                print(f"Error: {error.get('message', 'Unknown error')}")
                return

            elif event_type == "response.text.done":
                text = event.get("text", "")
                if text:
                    print(f"\nAssistant: {text}\nYou: ", end='', flush=True)

            elif event_type == "conversation.item.input_audio_transcription.completed":
                text = event.get("transcript", "")
                if text:
                    print(f"\nHeard: {text}", end='', flush=True)

            elif event_type == "session.created":
                print("Session started.")

            else:
                self.log(f"Unhandled event type: {event_type} {message}")

        except json.JSONDecodeError:
            print("Received non-JSON message.")
        except Exception as e:
            print(f"Exception in on_message: {e}")

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        self.log(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed.")
        self.log(f"WebSocket closed with code {close_status_code}, message: {close_msg}")
        if self.audio_sender:
            self.audio_sender.stop()

    def send_user_message(self, message):
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": message
                    }
                ]
            }
        }
        try:
            self.ws.send(json.dumps(event))
            print("Sending your message...")
            self.log(f"Sent user message: {message}")
        except Exception as e:
            print(f"Failed to send message: {e}")
            self.log(f"Failed to send message: {e}")

        # Send response.create to trigger the assistant's response
        response_create_event = {
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": "Please assist the user."
            }
        }
        try:
            self.ws.send(json.dumps(response_create_event))
            self.log("Sent response.create after user message.")
        except Exception as e:
            print(f"Failed to send response.create: {e}")
            self.log(f"Failed to send response.create: {e}")

    def run(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }

        self.ws = websocket.WebSocketApp(WEBSOCKET_URL,
                                         header=headers,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)

        # Run WebSocket in a separate thread
        wst = threading.Thread(target=self.ws.run_forever, daemon=True)
        wst.start()

        # Wait until the WebSocket connection is established
        while not self.ws.sock or not self.ws.sock.connected:
            pass

        print("Welcome to OpenAI Chat with Audio Streaming!")
        print("Type your messages below. Press Ctrl+C to exit.\n")

        while True:
            try:
                user_input = input("You: ")
                if user_input.strip().lower() in ["exit", "quit"]:
                    print("Exiting...")
                    break
                if user_input.strip() == "":
                    continue
                self.send_user_message(user_input)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

        self.ws.close()
        if self.audio_sender:
            self.audio_sender.stop()
        self.audio_player.close()

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    chat = ChatStreaming(api_key, verbose=False)
    chat.run()

if __name__ == "__main__":
    main()