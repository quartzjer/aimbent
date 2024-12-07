import json
import sys
import base64
import threading
import websocket
import pyaudio
import time

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4000  # 0.25 seconds worth of audio at 16kHz
FORMAT = pyaudio.paInt16

class ConvAIClient:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.ws = None
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_running = False

    def connect(self):
        url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={self.agent_id}"
        
        self.ws = websocket.WebSocketApp(
            url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        self.input_stream = self.audio.open(
            rate=SAMPLE_RATE,
            channels=CHANNELS,
            format=FORMAT,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        self.output_stream = self.audio.open(
            rate=SAMPLE_RATE,
            channels=CHANNELS,
            format=FORMAT,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        self.is_running = True

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            
            if data["type"] == "ping":
                ws.send(json.dumps({
                    "type": "pong",
                    "event_id": data["ping_event"]["event_id"]
                }))
            
            elif data["type"] == "audio":
                audio_data = base64.b64decode(data["audio_event"]["audio_base_64"])
                self.output_stream.write(audio_data)
            
            elif data["type"] == "user_transcript":
                print(f"You: {data['user_transcription_event']['user_transcript']}")
            
            elif data["type"] == "agent_response":
                print(f"Agent: {data['agent_response_event']['agent_response']}")
            
            else:
                print(f"Received unknown message: {data}")
                
        except Exception as e:
            print(f"Error processing message: {e}")

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")
        self.is_running = False

    def on_open(self, ws):
        print("WebSocket connection established")
        self.audio_thread = threading.Thread(target=self.send_audio, daemon=True)
        self.audio_thread.start()

    def send_audio(self):
        while self.is_running:
            try:
                audio_chunk = self.input_stream.read(CHUNK_SIZE)
                base64_audio = base64.b64encode(audio_chunk).decode('utf-8')
                self.ws.send(json.dumps({
                    "user_audio_chunk": base64_audio
                }))
                time.sleep(CHUNK_SIZE / SAMPLE_RATE)  # Control sending rate
            except Exception as e:
                print(f"Error sending audio: {e}")
                break

    def run(self):
        self.connect()
        self.ws.run_forever()

    def cleanup(self):
        self.is_running = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()
        if self.ws:
            self.ws.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <agent_id>")
        return

    client = ConvAIClient(sys.argv[1])
    try:
        client.run()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.cleanup()

if __name__ == "__main__":
    main()
