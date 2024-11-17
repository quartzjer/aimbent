import asyncio
import os
import sys
import logging
from pyrtmp.rtmp import SimpleRTMPController, RTMPProtocol, SimpleRTMPServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessingController(SimpleRTMPController):
    def __init__(self):
        super().__init__()
        self.ffmpeg_process = None
        self.ffmpeg_stdout_task = None
        self.ffmpeg_stderr_task = None

    async def start_ffmpeg(self):
        try:
            ffmpeg_cmd = (
                'ffmpeg -loglevel verbose '
                '-i pipe:0 '
                '-f s16le '
                '-acodec pcm_s16le '
                '-ac 1 '
                '-ar 24000 '
                '-strict experimental '
                '-max_muxing_queue_size 1024 '
                'pipe:1'
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
        """Read and log the PCM audio data from FFmpeg's stdout"""
        try:
            while True:
                data = await self.ffmpeg_process.stdout.read(4096)
                if not data:
                    break
                # Log information about the audio data chunk
                logger.info(f"Received PCM audio data chunk: {len(data)} bytes")
        except Exception as e:
            logger.error(f"Error reading FFmpeg output: {e}")

    async def read_ffmpeg_error(self):
        """Read and log any error output from FFmpeg"""
        try:
            while True:
                data = await self.ffmpeg_process.stderr.read(1024)
                if not data:
                    logger.info("FFmpeg error stream ended")
                    break
                logger.error(f"FFmpeg stderr: {data.decode()}")
        except Exception as e:
            logger.error(f"Error reading FFmpeg error: {e}")
            raise

    async def on_audio_message(self, session, message):
        """Handle incoming audio messages from RTMP stream"""
        try:
            if not self.ffmpeg_process or not self.ffmpeg_process.stdin:
                logger.error("FFmpeg process not available for audio message")
                return

            if self.ffmpeg_process.stdin.is_closing():
                logger.error("FFmpeg stdin is closing or closed")
                return

            logger.debug(f"Processing audio message of {len(message.payload)} bytes")
            self.ffmpeg_process.stdin.write(message.payload)
            #await self.ffmpeg_process.stdin.drain()
        except Exception as e:
            logger.error(f"Error processing audio message: {e}")
        await super().on_audio_message(session, message)

    async def on_stream_closed(self, session, exception):
        """Clean up FFmpeg when the stream is closed"""
        await self.cleanup_ffmpeg()
        await super().on_stream_closed(session, exception)

    async def on_ns_publish(self, session, message):
        """Start FFmpeg when a new stream is published"""
        await self.start_ffmpeg()
        await super().on_ns_publish(session, message)

class RTMPServer(SimpleRTMPServer):
    async def create(self, host, port):
        loop = asyncio.get_event_loop()
        self.server = await loop.create_server(
            lambda: RTMPProtocol(controller=AudioProcessingController()),
            host=host,
            port=port,
        )
        logger.info(f"RTMP server listening on {host}:{port}")

    async def start(self):
        async with self.server:
            await self.server.serve_forever()

async def main():
    try:
        server = RTMPServer()
        await server.create(host='0.0.0.0', port=1935)
        await server.start()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        logger.info("Cleaned up resources")

def run():
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run()