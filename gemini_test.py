import os
import asyncio
import json
import pyaudio
import websockets
from dotenv import load_dotenv
import logging
import signal
import sys
import queue
import base64

# Configure logging and load environment variables
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment.")

WS_ENDPOINT = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={API_KEY}"

class GeminiAudioClient:
    def __init__(self):
        self.stop_event = asyncio.Event()
        self.audio_input_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        
        # Input stream (microphone)
        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        # Output stream (speakers)
        self.output_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=1024
        )

    def build_setup_message(self):
        """Build initial setup message for websocket connection."""
        return {
            "setup": {
                "model": "models/gemini-2.0-flash-exp",
                "generationConfig": {
                    "responseModalities": "audio",
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": "Aoede"
                            }
                        }
                    }
                },
                "systemInstruction": {
                    "parts": [{"text": "You are my helpful assistant."}]
                },
                "tools": [
                    {"googleSearch": {}},
                    {
                        "functionDeclarations": [{
                            "name": "foo",
                            "description": "Does foo.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "bar": {
                                        "type": "string",
                                        "description": "bar stuff"
                                    }
                                },
                                "required": ["bar"]
                            }
                        }]
                    }
                ]
            }
        }

    async def read_microphone_audio(self):
        """Read audio data from microphone."""
        loop = asyncio.get_event_loop()
        while not self.stop_event.is_set():
            try:
                data = await loop.run_in_executor(None, self.input_stream.read, 1024, False)
                chunk = {
                    "mimeType": "audio/pcm;rate=16000",
                    "data": base64.b64encode(data).decode('utf-8')
                }
                self.audio_input_queue.put(chunk)
            except Exception as e:
                logging.error(f"Error reading microphone: {e}")
                await asyncio.sleep(0.1)

    async def send_audio_to_server(self, websocket):
        """Send audio data to server."""
        while not self.stop_event.is_set():
            try:
                if not self.audio_input_queue.empty():
                    chunk = self.audio_input_queue.get_nowait()
                    message = {
                        "realtimeInput": {
                            "mediaChunks": [chunk]
                        }
                    }
                    await websocket.send(json.dumps(message))
                    await asyncio.sleep(0.05)  # Rate limiting to prevent overwhelming
                else:
                    await asyncio.sleep(0.01)
            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                logging.error(f"Error sending audio: {e}")

    async def receive_audio_from_server(self, websocket):
        """Receive and process server messages."""
        async for message in websocket:
            if self.stop_event.is_set():
                break
            
            try:
                response = json.loads(message)
                
                if "setupComplete" in response:
                    logging.info("Setup complete")
                    continue
                
                if "serverContent" in response:
                    content = response["serverContent"]
                    if "modelTurn" in content:
                        for part in content["modelTurn"].get("parts", []):
                            if "inlineData" in part:
                                data = part["inlineData"]
                                if data["mimeType"] == "audio/pcm;rate=24000":
                                    audio_data = base64.b64decode(data["data"])
                                    self.output_stream.write(audio_data)
                    else:
                        logging.info(f"Received unknown server content: {response}")
                else:
                    logging.info(f"Received unknown message: {response}")
                
            except Exception as e:
                logging.error(f"Error processing server message: {e}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'input_stream'):
            self.input_stream.stop_stream()
            self.input_stream.close()
        if hasattr(self, 'output_stream'):
            self.output_stream.stop_stream()
            self.output_stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()

    async def run(self):
        """Main run loop."""
        try:
            async with websockets.connect(WS_ENDPOINT) as websocket:
                # Send setup message
                await websocket.send(json.dumps(self.build_setup_message()))
                
                # Create tasks
                tasks = [
                    asyncio.create_task(self.read_microphone_audio()),
                    asyncio.create_task(self.send_audio_to_server(websocket)),
                    asyncio.create_task(self.receive_audio_from_server(websocket))
                ]
                
                await self.stop_event.wait()
                
                # Cleanup tasks
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        finally:
            self.cleanup()

def main():
    client = GeminiAudioClient()
    
    def signal_handler(signum, frame):
        logging.info("Shutdown signal received")
        client.stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        client.cleanup()
        sys.exit(0)

if __name__ == "__main__":
    main()