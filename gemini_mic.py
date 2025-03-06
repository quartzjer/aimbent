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
from pyaec import Aec
from scipy.fft import rfft, irfft

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
        self.queue = Queue()
        self.is_suppressed = False
        self.last_active_time = None

    def queue_chunk(self, audio_data):
        self.queue.put(audio_data)

    def append_audio(self, audio_data):
        self.buffer = np.concatenate((self.buffer, audio_data))

    def prepend_audio(self, audio_data):
        self.buffer = np.concatenate((audio_data, self.buffer)) if self.buffer.size > 0 else audio_data

    def get_and_clear_buffer(self):
        if self.buffer.size == 0:
            return None
        snapshot = self.buffer.copy()
        self.buffer = np.array([], dtype=np.float32)
        return snapshot

class VoiceEnhancer:
    def __init__(self, sample_rate=16000, min_freq=300, max_freq=3400, boost_factor=1.5):
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.boost_factor = boost_factor
    
    def process(self, audio):
        if len(audio) == 0:
            return audio

        # clean edges
        audio = np.nan_to_num(audio, nan=0.0, posinf=1e10, neginf=-1e10)
        audio = np.where(audio == 0, 1e-10, audio)

        # Apply noise reduction
        audio = reduce_noise(y=audio, sr=self.sample_rate)

        # Compute the FFT
        X = rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), d=1/self.sample_rate)
        
        # Define vocal range mask
        vocal_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        
        # Apply boost to the vocal range frequencies
        X[vocal_mask] *= self.boost_factor
        
        # Reconstruct the time-domain signal via inverse FFT
        enhanced = irfft(X)
        
        # Return the enhanced audio as float32 in the same range as input
        return enhanced.astype(np.float32)

class AudioRecorder:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir or os.getcwd()
        self.model = load_silero_vad()
        self.client = genai.Client(api_key=API_KEY)
        self.devices = self._initialize_devices()
        self._running = True
        self.enhancer = VoiceEnhancer(sample_rate=SAMPLE_RATE, boost_factor=2)
        
        # PyAEC parameters - moved from process_chunks
        self.frame_size = int(0.02 * SAMPLE_RATE)
        self.filter_length = int(SAMPLE_RATE * 0.2)
        self.aec = Aec(self.frame_size, self.filter_length, SAMPLE_RATE, True)

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

    def record_from_device(self, device_key):
        """Record from a specific device using non-blocking pattern."""
        device = self.devices[device_key]
        print(f"Starting recording thread for {device.label}: {device.device.name}")
        
        try:
            with device.device.recorder(samplerate=SAMPLE_RATE, channels=[-1]) as recorder:
                while self._running:
                    try:
                        recording = recorder.record(numframes=None)
                        
                        if recording is not None and recording.size > 0:
                            device.queue_chunk(recording)
                            #print(f"Queued {len(recording) / SAMPLE_RATE:.4f} seconds from {device.label}")
                    except Exception as e:
                        print(f"Error recording from {device.label}: {e}")
                        if not self._running:
                            break
                        time.sleep(0.5)
        except Exception as e:
            print(f"Error setting up recorder for {device.label}: {e}")
            if self._running:  # Only log if not due to shutdown
                print(f"Recording thread for {device.label} crashed: {e}")

    def process_buffer(self, device):
        buffer_data = device.get_and_clear_buffer()
        if buffer_data is None:
            return []
            
        speech_segments = get_speech_timestamps(
            buffer_data, self.model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            speech_pad_ms=50,
            min_silence_duration_ms=100,
            min_speech_duration_ms=200,
            threshold=0.3
        )
        buffer_seconds = len(buffer_data) / SAMPLE_RATE
        print(f"Detected {len(speech_segments)} speech segments in {device.label} of {buffer_seconds:.1f} seconds.")
        # Debug: Save buffer to file
        debug_filename = f"test_{device.label}.ogg"
        debug_data = (np.clip(buffer_data, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(debug_filename, debug_data, SAMPLE_RATE, format='OGG', subtype='VORBIS')
        print(f"Saved debug file: {debug_filename}")

        segments = []
        total_duration = len(buffer_data) / SAMPLE_RATE

        for i, seg in enumerate(speech_segments):
            # If the last segment is too close to the end the speaking might be continuing
            if i == len(speech_segments) - 1 and total_duration - seg['end'] < 1:
                start_idx = int(seg['start'] * SAMPLE_RATE)
                unprocessed = buffer_data[start_idx:]
                device.prepend_audio(unprocessed)
                print(f"Unprocessed segment at end of {device.label} buffer of length {len(unprocessed)/SAMPLE_RATE:.1f} seconds.")
                break
            start_idx = int(seg['start'] * SAMPLE_RATE)
            end_idx = int(seg['end'] * SAMPLE_RATE)
            seg_data = buffer_data[start_idx:end_idx]
            segments.append({
                "offset": seg['start'],  # Store offset in seconds from buffer start
                "data": seg_data
            })
        
        return segments

    def transcribe(self, ogg_bytes):
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

    def calculate_rms(self, audio_buffer):
        """Calculate the Root Mean Square (RMS) of an audio buffer."""
        if len(audio_buffer) == 0:
            return 0
        return np.sqrt(np.mean(np.square(audio_buffer)))
    
    def normalize_audio(self, audio, target_rms):
        """Normalize audio to match a target RMS value."""
        if len(audio) == 0 or target_rms == 0:
            return audio
            
        current_rms = self.calculate_rms(audio)
        if current_rms == 0:
            return audio
            
        gain_factor = target_rms / current_rms
        return audio * gain_factor

    def process_chunks(self):
        mic_device = self.devices['mic']
        sys_device = self.devices['sys']
        
        # Empty both queues completely into full buffers
        mic_buffer = np.array([], dtype=np.float32)
        sys_buffer = np.array([], dtype=np.float32)
        
        # Collect all audio from system queue
        while not sys_device.queue.empty():
            chunk = sys_device.queue.get()
            sys_buffer = np.concatenate((sys_buffer, chunk.squeeze(axis=1)))

        # Collect all audio from mic queue
        while not mic_device.queue.empty():
            chunk = mic_device.queue.get()
            mic_buffer = np.concatenate((mic_buffer, chunk.squeeze(axis=1)))
        
        # Make sure both buffers have the same length
        min_length = min(len(mic_buffer), len(sys_buffer))
        if min_length == 0:
            return  # No data to process

        print(f"mic seconds {len(mic_buffer)/SAMPLE_RATE:.4f} sys seconds {len(sys_buffer)/SAMPLE_RATE:.4f}")        

        # Save original microphone audio to a file
        if mic_buffer.size > 0:
            orig_mic_int16 = (np.clip(mic_buffer, -1.0, 1.0) * 32767).astype(np.int16)
            sf.write("test_orig.ogg", orig_mic_int16, SAMPLE_RATE, format='OGG', subtype='VORBIS')
            print(f"Saved original microphone audio to test_orig.ogg ({len(orig_mic_int16)/SAMPLE_RATE:.2f} seconds)")

        # Convert float32 arrays to int16 (required by PyAEC)
        mic_int16 = (mic_buffer * 32767).astype(np.int16)
        sys_int16 = (sys_buffer * 32767).astype(np.int16)
        min_length = min(len(mic_int16), len(sys_int16))

        # Process in chunks of frame_size
        processed_chunks = []

        # Process complete frames
        for i in range(0, min_length, self.frame_size):
            # Get the frame (could be partial at the end)
            mic_frame = mic_int16[i:i+self.frame_size]
            sys_frame = sys_int16[i:i+self.frame_size]
            
            # If we have a partial frame at the end
            if min(len(mic_frame), len(sys_frame)) < self.frame_size:
                # Just append the raw partial frame
                processed_chunks.append(mic_frame)
            else:
                # Process with echo canceller
                processed_frame = self.aec.cancel_echo(mic_frame, sys_frame)
                processed_chunks.append(processed_frame)

        processed_mic = np.concatenate(processed_chunks).astype(np.float32) / 32767.0

        # Noise reduce and apply voice enhancer to mic after echo cancellation
        #processed_mic = reduce_noise(y=processed_mic, sr=SAMPLE_RATE)
        #processed_mic = self.enhancer.process(processed_mic)
        
        # Calculate RMS of system audio and normalize mic to match it
        sys_rms = self.calculate_rms(sys_buffer)
        if sys_rms > 0:
            processed_mic = self.normalize_audio(processed_mic, sys_rms)
            print(f"Normalized microphone audio to match system RMS: {sys_rms:.6f}")

        # Append processed audio to device buffers
        mic_device.append_audio(processed_mic)
        sys_device.append_audio(sys_buffer)
        
        print(f"Processed {len(processed_mic)/SAMPLE_RATE:.2f} seconds of audio through AEC and enhancement")

    def combine_speech_chunks_timer(self):
        while self._running:
            time.sleep(20)
            self.process_chunks()
            
            segments_all = []
            for _, device in self.devices.items():
                segments = self.process_buffer(device)
                segments_all.extend(segments)
                print(f"Processed {len(segments)} segments from {device.label}.")

            if not segments_all:
                continue

            # Sort segments based only on time offset
            segments_all.sort(key=lambda seg: seg["offset"])
            combined = np.concatenate([seg["data"] for seg in segments_all])
            chunk_int16 = (np.clip(combined, -1.0, 1.0) * 32767).astype(np.int16)
            
            buf = io.BytesIO()
            audio_data = chunk_int16.reshape(-1, CHANNELS)
            sf.write(buf, audio_data, SAMPLE_RATE, format='OGG', subtype='VORBIS')
            ogg_bytes = buf.getvalue()
            
            response_text = self.transcribe(ogg_bytes)
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
            threading.Thread(target=self.record_from_device, args=('mic',), daemon=True),
            threading.Thread(target=self.record_from_device, args=('sys',), daemon=True),
            threading.Thread(target=self.combine_speech_chunks_timer, daemon=True)
        ]
        for thread in threads:
            thread.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self._running = False
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
