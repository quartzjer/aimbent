import os
import sys
import io
import numpy as np
import soundcard as sc
import soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps
from dotenv import load_dotenv
from google import genai
from google.genai import types
import threading
import datetime
import json
from noisereduce import reduce_noise
import time
from queue import Queue

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit("Error: GOOGLE_API_KEY not found in environment. Please set it in your .env file.")

# Check for an optional CLI argument for the save directory.
if len(sys.argv) > 1:
    save_dir = sys.argv[1]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
else:
    save_dir = os.getcwd()

CHUNK_DURATION = 5
SAMPLE_RATE = 16000
CHANNELS = 1

class AudioDevice:
    def __init__(self, device, label):
        self.device = device
        self.label = label
        self.buffer = np.array([], dtype=np.float32)
        self.timestamp = None
        self.queue = Queue()
        self.is_suppressed = False
        self.last_active_time = None

    def queue_chunk(self, audio_data, timestamp):
        self.queue.put((audio_data, timestamp))

    def append_audio(self, audio_data, timestamp=None):
        if self.timestamp is None:
            self.timestamp = timestamp or datetime.datetime.now()
        self.buffer = np.concatenate((self.buffer, audio_data))

    def prepend_audio(self, audio_data, timestamp):
        self.buffer = np.concatenate((audio_data, self.buffer)) if self.buffer.size > 0 else audio_data
        self.timestamp = timestamp

    def get_and_clear_buffer(self):
        if self.buffer.size == 0 or self.timestamp is None:
            return None
        snapshot = (self.buffer.copy(), self.timestamp)
        self.buffer = np.array([], dtype=np.float32)
        self.timestamp = None
        return snapshot

class AudioRecorder:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir or os.getcwd()
        self.model = load_silero_vad()
        self.client = genai.Client(api_key=API_KEY)
        self.devices = self._initialize_devices()
        self._running = True
        self.record_barrier = threading.Barrier(2)

    def _initialize_devices(self):
        mics = sc.all_microphones(include_loopback=True)
        if len(mics) < 2:
            raise RuntimeError("At least 2 audio input devices are required!")
        print("Using devices:")
        print(f"Microphone: {mics[0].name}")
        print(f"System audio: {mics[1].name}")
        return {
            'mic': AudioDevice(mics[0], "microphone"),
            'sys': AudioDevice(mics[1], "system")
        }

    def record_device(self, device_key):
        device = self.devices[device_key]
        while self._running:
            try:
                self.record_barrier.wait(timeout=5)
                
                recording = device.device.record(
                    samplerate=SAMPLE_RATE,
                    numframes=CHUNK_DURATION * SAMPLE_RATE,
                    channels=[0]
                )
                # Queue the recorded chunk
                device.queue_chunk(recording.squeeze(axis=1), datetime.datetime.now())
                print(f"Queued {CHUNK_DURATION} seconds from {device.device.name}")
            except threading.BrokenBarrierError:
                if self._running:
                    print(f"Synchronization broken for {device.label}")
                break
            except Exception as e:
                print(f"Error recording from {device.label}: {e}")
                if self._running:
                    self.record_barrier.reset()
                break

    def process_buffer(self, device):
        snapshot = device.get_and_clear_buffer()
        if not snapshot:
            return []
            
        buffer_data, base_ts = snapshot
        speech_segments = get_speech_timestamps(
            buffer_data, self.model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            speech_pad_ms=100,
            min_silence_duration_ms=500
        )
        
        segments = []
        total_duration = len(buffer_data) / SAMPLE_RATE

        for seg in speech_segments:
            if total_duration - seg['end'] < 1:
                start_idx = int(seg['start'] * SAMPLE_RATE)
                unprocessed = buffer_data[start_idx:]
                unproc_ts = base_ts + datetime.timedelta(seconds=seg['start'])
                device.prepend_audio(unprocessed, unproc_ts)
                break
            start_idx = int(seg['start'] * SAMPLE_RATE)
            end_idx = int(seg['end'] * SAMPLE_RATE)
            seg_data = buffer_data[start_idx:end_idx]
            seg_abs_ts = base_ts + datetime.timedelta(seconds=seg['start'])
            segments.append({"timestamp": seg_abs_ts, "data": seg_data})
        
        return segments

    def process_audio_chunk(self, ogg_bytes):
        size_mb = len(ogg_bytes) / (1024 * 1024)
        print(f"Transcribing chunk: {size_mb:.2f}MB")
        
        with open("gemini_mic.txt", "r") as f:
            prompt_text = f.read().strip()
            
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    "Process the provided audio now and output your professional accurate transcription in the specified JSON format.",
                    types.Part.from_bytes(data=ogg_bytes, mime_type="audio/ogg")
                ],
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    system_instruction=prompt_text
                )
            )
            return response.text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    def combine_speech_chunks_timer(self):
        while self._running:
            time.sleep(20)
            
            segments_all = []
            for device_key, device in self.devices.items():
                segments = self.process_buffer(device)
                segments_all.extend(segments)
                print(f"Processed {len(segments)} segments from {device.label}.")

            if not segments_all:
                continue

            segments_all.sort(key=lambda x: x["timestamp"])
            combined = np.concatenate([seg["data"] for seg in segments_all])
            combined_clean = reduce_noise(y=combined, sr=SAMPLE_RATE)
            chunk_int16 = (np.clip(combined_clean, -1.0, 1.0) * 32767).astype(np.int16)
            
            buf = io.BytesIO()
            audio_data = chunk_int16.reshape(-1, CHANNELS)
            sf.write(buf, audio_data, SAMPLE_RATE, format='OGG', subtype='VORBIS')
            ogg_bytes = buf.getvalue()
            
            response_text = self.process_audio_chunk(ogg_bytes)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.save_results(timestamp, ogg_bytes, response_text)

    def save_results(self, timestamp, ogg_bytes, response_text):
        ogg_filepath = os.path.join(self.save_dir, f"audio_{timestamp}.ogg")
        json_filepath = os.path.join(self.save_dir, f"audio_{timestamp}.json")
        
        with open(ogg_filepath, "wb") as f:
            f.write(ogg_bytes)
        with open(json_filepath, "w") as f:
            json.dump({"text": response_text}, f)
        print(f"Saved to {ogg_filepath}: {response_text}")

    def start(self):
        threads = [
            threading.Thread(target=self.record_device, args=('mic',), daemon=True),
            threading.Thread(target=self.record_device, args=('sys',), daemon=True),
            threading.Thread(target=self.combine_speech_chunks_timer, daemon=True)
        ]
        for thread in threads:
            thread.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self._running = False
            self.record_barrier.reset()  # Break the barrier to allow threads to exit
            print("\nRecording stopped (Ctrl+C pressed)")
        except Exception as e:
            self._running = False
            print(f"Error during recording: {e}")

def main():
    save_dir = sys.argv[1] if len(sys.argv) > 1 else None
    recorder = AudioRecorder(save_dir)
    recorder.start()

if __name__ == "__main__":
    main()
