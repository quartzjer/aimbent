import asyncio
import os
import sys
import json
import logging
import base64
import websocket
from threading import Thread
from pyrtmp.rtmp import SimpleRTMPController, RTMPProtocol, SimpleRTMPServer
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

class OpenAIWebSocketClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.connected = False
        self.ws = None
        self.send_queue = asyncio.Queue()
        self.ws_thread = None
        self.session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad", "threshold": 0.8},
                "temperature": 0.7,
                "max_response_output_tokens": 500
            }
        }

    def on_message(self, ws, message):
        asyncio.run_coroutine_threadsafe(self.handle_message(message), asyncio.get_event_loop())

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")
        self.connected = False

    def on_open(self, ws):
        logger.info("WebSocket connected")
        self.connected = True
        ws.send(json.dumps(self.session_update))

    def websocket_run(self):
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            WEBSOCKET_URL,
            header={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            },
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever()

    async def connect(self):
        try:
            self.ws_thread = Thread(target=self.websocket_run, daemon=True)
            self.ws_thread.start()
            # Wait for connection to be established
            for _ in range(10):  # timeout after 5 seconds
                if self.connected:
                    break
                await asyncio.sleep(0.5)
            if not self.connected:
                raise ConnectionError("Failed to connect to OpenAI WebSocket")
            asyncio.create_task(self.send_messages())
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise

    async def handle_message(self, message):
        try:
            event = json.loads(message)
            logger.info(f"Received event type: {event.get('type')}")
            # Handle different event types as needed
        except json.JSONDecodeError:
            logger.error("Received non-JSON message")

    async def send_messages(self):
        while self.connected:
            data = await self.send_queue.get()
            try:
                if self.ws and self.ws.sock:
                    self.ws.send(json.dumps(data))
                    logger.info(f"Sent audio data length {len(data.get('audio', ''))}")
            except Exception as e:
                logger.error(f"Error sending audio data: {e}")

    async def send_audio_data(self, data):
        if self.connected:
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(data).decode('utf-8')
            }
            await self.send_queue.put(audio_event)
        else:
            logger.error("WebSocket is not connected")
    
    async def close(self):
        if self.ws:
            self.ws.close()
        self.connected = False
        if self.ws_thread:
            self.ws_thread.join(timeout=5.0)

class RTMP2OpenAIController(SimpleRTMPController):
    def __init__(self, websocket_client):
        super().__init__()
        self.websocket_client = websocket_client
        self.ffmpeg_process = None
        self.ffmpeg_stdout_task = None
        self.ffmpeg_stderr_task = None

    async def start_ffmpeg(self):
        try:
            ffmpeg_cmd = (
                'ffmpeg -loglevel warning '
                '-f aac -i pipe:0 '  # Read AAC from stdin
                '-f s16le '          # Output format
                '-acodec pcm_s16le ' # Output codec
                '-ac 1 '             # Mono audio
                '-ar 24000 '         # Sample rate
                'pipe:1'             # Output to stdout
            )
            self.ffmpeg_process = await asyncio.create_subprocess_shell(
                ffmpeg_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self.ffmpeg_stdout_task = asyncio.create_task(self.read_ffmpeg_output())
            self.ffmpeg_stderr_task = asyncio.create_task(self.read_ffmpeg_error())
            logger.info("FFmpeg subprocess started successfully")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            await self.cleanup_ffmpeg()
            raise

    async def cleanup_ffmpeg(self):
        """Helper method to clean up FFmpeg resources"""
        if self.ffmpeg_stdout_task and not self.ffmpeg_stdout_task.done():
            self.ffmpeg_stdout_task.cancel()
            try:
                await self.ffmpeg_stdout_task
            except asyncio.CancelledError:
                pass

        if self.ffmpeg_process:
            try:
                if self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.is_closing():
                    self.ffmpeg_process.stdin.write_eof()
                    await self.ffmpeg_process.stdin.drain()
            except Exception as e:
                logger.error(f"Error closing FFmpeg stdin: {e}")
            
            try:
                # Read any remaining stderr
                stderr = await self.ffmpeg_process.stderr.read()
                if stderr:
                    logger.error(f"FFmpeg stderr: {stderr.decode()}")
                
                # Give the process a chance to exit gracefully
                try:
                    await asyncio.wait_for(self.ffmpeg_process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg process didn't exit gracefully, terminating")
                    self.ffmpeg_process.terminate()
                    await self.ffmpeg_process.wait()
            except Exception as e:
                logger.error(f"Error during FFmpeg cleanup: {e}")
            finally:
                self.ffmpeg_process = None

    async def read_ffmpeg_output(self):
        try:
            while True:
                data = await self.ffmpeg_process.stdout.read(4096)  # Increased buffer size
                if not data:
                    break
                await self.websocket_client.send_audio_data(data)
        except Exception as e:
            logger.error(f"Error reading FFmpeg output: {e}")

    async def read_ffmpeg_error(self):
        try:
            while True:
                data = await self.ffmpeg_process.stderr.read(1024)
                logger.error(f"FFmpeg stderr: {data.decode()}")
                if not data:
                    logger.info("FFmpeg error stream ended")
                    break

        except Exception as e:
            logger.error(f"Error reading FFmpeg error: {e}")
            raise

    async def on_audio_message(self, session, message):
        try:
            if not self.ffmpeg_process or not self.ffmpeg_process.stdin:
                logger.error("FFmpeg process not available for audio message")
                return

            if self.ffmpeg_process.stdin.is_closing():
                logger.error("FFmpeg stdin is closing or closed")
                return

            logger.debug(f"Processing audio message of {len(message.payload)} bytes")
            self.ffmpeg_process.stdin.write(message.payload)
            await self.ffmpeg_process.stdin.drain()
        except Exception as e:
            logger.error(f"Error processing audio message: {e}")
        await super().on_audio_message(session, message)

    async def on_stream_closed(self, session, exception):
        await self.cleanup_ffmpeg()
        await super().on_stream_closed(session, exception)

    async def on_ns_publish(self, session, message):
        await self.start_ffmpeg()
        await super().on_ns_publish(session, message)

class SimpleServer(SimpleRTMPServer):
    def __init__(self, websocket_client):
        self.websocket_client = websocket_client
        super().__init__()

    async def create(self, host, port):
        loop = asyncio.get_event_loop()
        self.server = await loop.create_server(
            lambda: RTMPProtocol(controller=RTMP2OpenAIController(self.websocket_client)),
            host=host,
            port=port,
        )
        logger.info(f"RTMP server listening on {host}:{port}")

    async def start(self):
        async with self.server:
            await self.server.serve_forever()

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    websocket_client = OpenAIWebSocketClient(api_key)
    
    try:
        await websocket_client.connect()

        server = SimpleServer(websocket_client)
        await server.create(host='0.0.0.0', port=1935)
        await server.start()  # This will block until the server is stopped

    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        await websocket_client.close()
        logger.info("Cleaned up resources")

def run():
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run()