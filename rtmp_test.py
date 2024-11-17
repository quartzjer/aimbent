import asyncio
import logging
from dataclasses import dataclass
import pyaudio
import threading
from queue import Queue
import av

from pyrtmp import StreamClosedException
from pyrtmp.session_manager import SessionManager
from pyrtmp.rtmp import SimpleRTMPController, RTMPProtocol, SimpleRTMPServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio format constants
@dataclass
class AudioConfig:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 24000
    chunk_size: int = 1024
    layout: str = 'mono'

# RTMP Sound format mapping
SOUNDFORMAT_CODECS = {
    10: 'aac',
    2: 'mp3',
    1: 'adpcm',
    0: 'pcm',
}

class AudioPlayer:
    def __init__(self, config: AudioConfig = AudioConfig()):
        self.config = config
        self.audio_queue: Queue = Queue()
        self.running: bool = True
        
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=config.format,
            channels=config.channels,
            rate=config.rate,
            output=True,
            frames_per_buffer=config.chunk_size
        )

        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def _playback_loop(self) -> None:
        while self.running:
            try:
                data = self.audio_queue.get()
                if data:  # Check for non-empty data
                    self.stream.write(data)
            except Exception as e:
                logger.error(f"Playback error: {e}")

    def write(self, frame: av.AudioFrame) -> None:
        try:
            # Get raw bytes from the audio frame
            pcm_data = bytes(frame.planes[0])
            self.audio_queue.put(pcm_data)
        except Exception as e:
            logger.error(f"Audio write error: {e}")

    def close(self) -> None:
        self.running = False
        self.playback_thread.join(timeout=1)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pa:
            self.pa.terminate()

class RTMPAudioController(SimpleRTMPController):

    async def on_ns_publish(self, session: SessionManager, message) -> None:
        session.state = AudioPlayer()
        await super().on_ns_publish(session, message)

    async def on_audio_message(self, session: SessionManager, message) -> None:
        try:
            payload = message.payload
            if not payload:
                return

            # Parse audio format
            sound_format = (payload[0] & 0xF0) >> 4
            codec_name = SOUNDFORMAT_CODECS.get(sound_format)
            if not codec_name:
                logger.error(f"Unsupported sound format: {sound_format}")
                return

            # Initialize decoder and resampler if needed
            if not hasattr(session.state, 'decoder'):
                session.state.decoder = av.codec.CodecContext.create(codec_name, 'r')
                session.state.resampler = av.AudioResampler(
                    format='s16',
                    layout='mono',
                    rate=AudioConfig.rate
                )

            # Handle codec-specific processing
            if codec_name == 'aac':
                if payload[1] == 0:  # AAC sequence header
                    session.state.decoder.extradata = payload[2:]
                    return
                payload = payload[2:]  # Skip AAC packet type
            else:
                payload = payload[1:]  # Skip sound format byte

            packet = av.packet.Packet(payload)
            frames = session.state.decoder.decode(packet)

            for frame in frames:
                resampled_frame = session.state.resampler.resample(frame)[0]  # Get first frame
                if resampled_frame:
                    session.state.write(resampled_frame)

        except Exception as e:
            logger.error(f"Audio processing error: {e}", exc_info=True)

        await super().on_audio_message(session, message)

    async def on_stream_closed(self, session: SessionManager, exception: StreamClosedException) -> None:
        if session.state:
            session.state.close()
        await super().on_stream_closed(session, exception)


class RTMPServer(SimpleRTMPServer):

    async def create(self, host: str, port: int) -> None:
        self.server = await asyncio.get_event_loop().create_server(
            lambda: RTMPProtocol(controller=RTMPAudioController()),
            host=host,
            port=port
        )


async def main():
    try:
        server = RTMPServer()
        await server.create(host='0.0.0.0', port=1935)
        logger.info("RTMP server started on port 1935")
        await server.start()
        await server.wait_closed()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())