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
import subprocess
import argparse
from audio_detect import audio_detect

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

class AudioRecorder:
    def __init__(self, save_dir=None, debug=False, timer_interval=60):
        self.save_dir = save_dir or os.getcwd()
        self.model = load_silero_vad()
        self.client = genai.Client(api_key=API_KEY)
        # Initialize devices and separate queues for mic and system audio
        self.mic_device, self.sys_device = self._initialize_devices()
        self.mic_queue = Queue()
        self.sys_queue = Queue()
        self._running = True
        self.debug = debug
        
        # Voice enhancement parameters
        self.min_freq = 300
        self.max_freq = 3400
        self.boost_factor = 2
        
        # PyAEC parameters
        self.frame_size = int(0.02 * SAMPLE_RATE)
        self.filter_length = int(SAMPLE_RATE * 0.2)
        self.aec = Aec(self.frame_size, self.filter_length, SAMPLE_RATE, True)
        self.timer_interval = timer_interval

    def enhance_voice(self, audio):
        if len(audio) == 0:
            return audio

        # clean edges
        audio = np.nan_to_num(audio, nan=0.0, posinf=1e10, neginf=-1e10)
        audio = np.where(audio == 0, 1e-10, audio)

        # Apply noise reduction
        audio = reduce_noise(y=audio, sr=SAMPLE_RATE)

        # Compute the FFT
        X = rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), d=1/SAMPLE_RATE)
        
        # Define vocal range mask
        vocal_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        
        # Apply boost to the vocal range frequencies
        X[vocal_mask] *= self.boost_factor
        
        # Reconstruct the time-domain signal via inverse FFT
        enhanced = irfft(X)
        
        # Return the enhanced audio as float32 in the same range as input
        return enhanced.astype(np.float32)

    def _initialize_devices(self):
        mic, loopback = audio_detect()
        if mic is None or loopback is None:
            raise RuntimeError("Failed to detect required audio devices!")
        print("Using devices:")
        print(f"Microphone: {mic.name}")
        print(f"System audio: {loopback.name}")
        return mic, loopback

    # New generic recording method accepting device, queue and label.
    def record_device(self, device, queue, label):
        print(f"Starting recording thread for {label}: {device.name}")
        try:
            with device.recorder(samplerate=SAMPLE_RATE, channels=[-1]) as recorder:
                while self._running:
                    try:
                        recording = recorder.record(numframes=None)
                        if recording is not None and recording.size > 0:
                            queue.put(recording)
                    except Exception as e:
                        print(f"Error recording from {label}: {e}")
                        if not self._running:
                            break
                        time.sleep(0.5)
        except Exception as e:
            print(f"Error setting up recorder for {label}: {e}")
            if self._running:
                print(f"Recording thread for {label} crashed: {e}")

    def detect_speech(self, label, buffer_data):
        if buffer_data is None or len(buffer_data) == 0:
            return [], np.array([], dtype=np.float32)
            
        speech_segments = get_speech_timestamps(
            buffer_data, self.model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            speech_pad_ms=70,
            min_silence_duration_ms=100,
            min_speech_duration_ms=200,
            threshold=0.3
        )
        buffer_seconds = len(buffer_data) / SAMPLE_RATE
        print(f"Detected {len(speech_segments)} speech segments in {label} of {buffer_seconds:.1f} seconds.")
        if self.debug:
            debug_filename = f"test_{label}.ogg"
            debug_data = (np.clip(buffer_data, -1.0, 1.0) * 32767).astype(np.int16)
            sf.write(debug_filename, debug_data, SAMPLE_RATE, format='OGG', subtype='VORBIS')
            print(f"Saved debug file: {debug_filename}")

        segments = []
        total_duration = len(buffer_data) / SAMPLE_RATE
        unprocessed_data = np.array([], dtype=np.float32)

        for i, seg in enumerate(speech_segments):
            if i == len(speech_segments) - 1 and total_duration - seg['end'] < 1:
                start_idx = int(seg['start'] * SAMPLE_RATE)
                unprocessed_data = buffer_data[start_idx:]
                print(f"Unprocessed segment at end of {label} buffer of length {len(unprocessed_data)/SAMPLE_RATE:.1f} seconds.")
                break
            start_idx = int(seg['start'] * SAMPLE_RATE)
            end_idx = int(seg['end'] * SAMPLE_RATE)
            seg_data = buffer_data[start_idx:end_idx]
            segments.append({
                "offset": seg['start'],
                "data": seg_data
            })
        return segments, unprocessed_data

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

    def is_system_muted(self):
        """Check if the system audio is muted."""
        try:
            result = subprocess.run(
                ["pactl", "get-sink-mute", "@DEFAULT_SINK@"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return "Mute: yes" in result.stdout
        except subprocess.SubprocessError as e:
            print(f"Error checking system mute status: {e}")
            return False

    def get_device_buffers(self):
        mic_buffer = np.array([], dtype=np.float32)
        sys_buffer = np.array([], dtype=np.float32)
        
        while not self.sys_queue.empty():
            chunk = self.sys_queue.get()
            sys_buffer = np.concatenate((sys_buffer, chunk.squeeze(axis=1)))

        while not self.mic_queue.empty():
            chunk = self.mic_queue.get()
            mic_buffer = np.concatenate((mic_buffer, chunk.squeeze(axis=1)))
        
        return mic_buffer, sys_buffer

    def apply_echo_cancellation(self, mic_buffer, sys_buffer):
        
        min_length = min(len(mic_buffer), len(sys_buffer))
        if min_length == 0:
            print("Missing audio data in one or both buffers.")
            return mic_buffer, sys_buffer

        sys_rms = self.calculate_rms(sys_buffer)
        if sys_rms < 0.01:
            print("System audio is silent.")
            return mic_buffer, sys_buffer

        print(f"echo cancelling mic seconds {len(mic_buffer)/SAMPLE_RATE:.4f} sys seconds {len(sys_buffer)/SAMPLE_RATE:.4f}")        

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
        
        # Only enhance/normalize if we're doing echo cancellation
        processed_mic = self.enhance_voice(processed_mic)
        processed_mic = self.normalize_audio(processed_mic, sys_rms)
        print(f"Normalized microphone audio to match system RMS: {sys_rms:.6f}")
        
        return processed_mic

    def process_segments_and_transcribe(self, segments, suffix=None):
        if not segments:
            return
            
        # Concatenate all segments
        combined = np.concatenate([seg["data"] for seg in segments])
        chunk_int16 = (np.clip(combined, -1.0, 1.0) * 32767).astype(np.int16)
        
        # Create OGG file
        buf = io.BytesIO()
        audio_data = chunk_int16.reshape(-1, CHANNELS)
        sf.write(buf, audio_data, SAMPLE_RATE, format='OGG', subtype='VORBIS')
        ogg_bytes = buf.getvalue()
        
        # Transcribe audio
        response_text = self.transcribe(ogg_bytes)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add suffix if provided
        if suffix:
            timestamp += suffix
            
        # Save results
        self.save_results(timestamp, ogg_bytes, response_text)

    def speech_processing_timer(self):
        mic_buffer = np.array([], dtype=np.float32)
        sys_buffer = np.array([], dtype=np.float32)
        while self._running:
            time.sleep(self.timer_interval)
            system_muted = self.is_system_muted()
            print(f"System audio mute status: {'Muted' if system_muted else 'Not muted'}")
            new_mic, new_sys = self.get_device_buffers()
            if system_muted:
                mic_buffer = np.concatenate((mic_buffer, new_mic)) if mic_buffer.size > 0 else new_mic
                sys_buffer = np.concatenate((sys_buffer, new_sys)) if sys_buffer.size > 0 else new_sys
                mic_segments, unprocessed_mic = self.detect_speech("mic", mic_buffer)
                if mic_segments:
                    self.process_segments_and_transcribe(mic_segments, suffix="_mic")
                    print(f"Found {len(mic_segments)} microphone segments")
                else:
                    print("No microphone segments found")
                sys_segments, unprocessed_sys = self.detect_speech("sys", sys_buffer)
                if sys_segments:
                    self.process_segments_and_transcribe(sys_segments, suffix="_sys")
                    print(f"Found {len(sys_segments)} system audio segments")
                else:
                    print("No system audio segments found")
                mic_buffer = unprocessed_mic
                sys_buffer = unprocessed_sys
            else:
                new_mic = self.apply_echo_cancellation(new_mic, new_sys)
                mic_buffer = np.concatenate((mic_buffer, new_mic)) if mic_buffer.size > 0 else new_mic
                sys_buffer = np.concatenate((sys_buffer, new_sys)) if sys_buffer.size > 0 else new_sys
                segments_all = []
                mic_segments, unprocessed_mic = self.detect_speech("mic", mic_buffer)
                sys_segments, unprocessed_sys = self.detect_speech("sys", sys_buffer)
                segments_all.extend(mic_segments)
                segments_all.extend(sys_segments)
                print(f"Found {len(mic_segments)} microphon and {len(sys_segments)} system segments.")
                if segments_all:
                    segments_all.sort(key=lambda seg: seg["offset"]) # weave together in time
                    self.process_segments_and_transcribe(segments_all)
                mic_buffer = unprocessed_mic
                sys_buffer = unprocessed_sys

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
            threading.Thread(target=self.record_device, args=(self.mic_device, self.mic_queue, "mic"), daemon=True),
            threading.Thread(target=self.record_device, args=(self.sys_device, self.sys_queue, "sys"), daemon=True),
            threading.Thread(target=self.speech_processing_timer, daemon=True)
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
    parser = argparse.ArgumentParser(description="Record audio and transcribe using Gemini API.")
    parser.add_argument("save_dir", nargs="?", default=None, help="Directory to save audio and transcriptions.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode (save audio buffers).")
    parser.add_argument("-t", "--timer_interval", type=int, default=60, help="Timer interval in seconds.")
    args = parser.parse_args()

    recorder = AudioRecorder(args.save_dir, args.debug, args.timer_interval)
    recorder.start()

if __name__ == "__main__":
    main()
