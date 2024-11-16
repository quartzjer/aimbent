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
import argparse
import wave
import time
import subprocess
from queue import Queue

load_dotenv()

WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024

interrupted = False

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("\nExiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        
        # Get default output device info
        default_device = self.p.get_default_output_device_info()
        print(f"Using audio output device: {default_device['name']}")
        
        try:
            self.stream = self.p.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                output_device_index=default_device['index'],
                frames_per_buffer=CHUNK
            )
            print("Audio playback initialized successfully")
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            print("Please check your audio output device settings")
            sys.exit(1)
        
        self.queue = Queue()
        self.running = True  # Set running before starting the thread
        self.playback_thread = threading.Thread(target=self.playback_loop, daemon=True)
        self.playback_thread.start()

    def playback_loop(self):
        while self.running:
            try:
                audio_bytes = self.queue.get(timeout=0.1)
                try:
                    self.stream.write(audio_bytes)
                except Exception as e:
                    print(f"Error playing audio: {e}")
            except:
                continue  # No audio data in queue, continue looping

    def play_audio(self, audio_bytes):
        self.queue.put(audio_bytes)

    def close(self):
        self.running = False
        self.playback_thread.join()
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        except Exception as e:
            print(f"Error closing audio stream: {e}")

class AudioSender:
    def __init__(self, ws, audio_file, audio_player):
        self.ws = ws
        self.audio_file = audio_file
        self.audio_player = audio_player
        self.thread = threading.Thread(target=self.send_audio, daemon=True)
        self.running = False
        self.process = None

    def start(self):
        self.running = True
        self.thread.start()

    def send_audio(self):
        try:
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', self.audio_file,
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ar', str(RATE),
                '-ac', str(CHANNELS),
                '-loglevel', 'error',  # Reduce ffmpeg output
                '-'
            ]
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # Suppress stderr
                stdin=subprocess.DEVNULL   # Ensure ffmpeg doesn't read from stdin
            )

            while self.running and not interrupted:
                data = self.process.stdout.read(CHUNK * 2)
                if not data:
                    break
                # Play the audio chunk
                self.audio_player.play_audio(data)
                # Send the audio chunk
                audio_base64 = base64.b64encode(data).decode('utf-8')
                audio_event = {"type": "input_audio_buffer.append", "audio": audio_base64}
                self.ws.send(json.dumps(audio_event))
                time.sleep(CHUNK / RATE)  # Control the sending rate

        except Exception as e:
            print(f"Error sending audio: {e}")
        finally:
            if self.process:
                self.process.terminate()
                self.process.wait()
            self.running = False

    def commit_buffer(self):
        commit_event = {"type": "input_audio_buffer.commit"}
        try:
            self.ws.send(json.dumps(commit_event))
        except Exception as e:
            print(f"Failed to commit audio buffer: {e}")

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.commit_buffer()

class ChatStreaming:
    def __init__(self, api_key, audio_file, verbose=False):
        self.api_key = api_key
        self.ws = None
        self.audio_player = AudioPlayer()
        self.audio_sender = None
        self.verbose = verbose
        self.audio_file = audio_file
        self.audio_buffer = BytesIO()

    def log(self, message):
        if self.verbose:
            print(f"[DEBUG] {message}")

    def on_open(self, ws):
        session_update_message = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": "You will be listening to a conversation, your job is to notice when you can be helpful. You are not part of the conversation, you are only an observer, and you don't need to transcribe, just focus on providing concise actionable suggestions.",
                "turn_detection": None,
                "temperature": 0.7,
                "max_response_output_tokens": 500
            }
        }
        ws.send(json.dumps(session_update_message))

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
                    print(f"Aimbe: {text}\n> ", end='', flush=True)

            elif event_type == "session.created":
                print("Session started.")
                # Start audio streaming after session is created
                self.audio_sender = AudioSender(ws, self.audio_file, self.audio_player)
                self.audio_sender.start()

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
                "content": [{"type": "input_text", "text": message}]
            }
        }
        try:
            self.ws.send(json.dumps(event))
            self.log(f"Sent user message: {message}")
            self.audio_sender.commit_buffer()
        except Exception as e:
            print(f"Failed to send message: {e}")
            self.log(f"Failed to send message: {e}")

        response_create_event = {
            "type": "response.create",
            "response": {"modalities": ["text"], "instructions": "Please assist the user."}
        }
        try:
            self.ws.send(json.dumps(response_create_event))
            self.log("Sent response.create after user message.")
        except Exception as e:
            print(f"Failed to send response.create: {e}")
            self.log(f"Failed to send response.create: {e}")

    def handle_user_input(self):
        while True:
            try:
                user_input = input("> ")
                if user_input.strip().lower() in ["exit", "quit"]:
                    print("Exiting...")
                    break
                if user_input.strip() == "":
                    continue
                self.send_user_message(user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

    def close(self):
        if self.ws:
            self.ws.close()
        if self.audio_sender:
            self.audio_sender.stop()
        if self.audio_player:
            self.audio_player.close()

    def run(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = websocket.WebSocketApp(
            WEBSOCKET_URL,
            header=headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        wst = threading.Thread(target=self.ws.run_forever, daemon=True)
        wst.start()

        try:
            self.handle_user_input()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
        finally:
            self.close()
            wst.join()

def main():
    parser = argparse.ArgumentParser(description="OpenAI Chat with Audio Streaming")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (.mp4) to stream")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    chat = ChatStreaming(api_key, args.audio_file, verbose=args.verbose)
    chat.run()

if __name__ == "__main__":
    main()