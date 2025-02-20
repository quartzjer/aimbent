import os
import sys
import io
import numpy as np
import sounddevice as sd
import soundfile as sf  # New dependency for OGG/Vorbis encoding
from silero_vad import load_silero_vad, get_speech_timestamps
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit("Error: GOOGLE_API_KEY not found in environment. Please set it in your .env file.")

if len(sys.argv) < 2:
    sys.exit("Usage: python gemini_mic.py <chunk_duration_seconds>")
CHUNK_DURATION = float(sys.argv[1])

SAMPLE_RATE = 16000
CHANNELS = 1

# Load Silero VAD model and initialize Gemini client
model = load_silero_vad()
client = genai.Client(api_key=API_KEY)

def process_audio_chunk(audio_chunk):
    """
    Encode the given audio chunk as OGG (using Vorbis via PySoundFile) and send it to Gemini for transcription.
    `audio_chunk` is expected to be an int16 NumPy array.
    """
    buf = io.BytesIO()
    # PySoundFile expects data in shape (samples, channels). Reshape for mono.
    audio_data = audio_chunk.reshape(-1, CHANNELS)
    # Write the OGG/Vorbis file into the BytesIO buffer.
    sf.write(buf, audio_data, SAMPLE_RATE, format='OGG', subtype='VORBIS')
    ogg_bytes = buf.getvalue()

    duration_sec = len(audio_chunk) / SAMPLE_RATE
    size_mb = len(ogg_bytes) / (1024 * 1024)
    print(f"Transcribing chunk: {duration_sec:.1f}s, {size_mb:.2f}MB")

    prompt_text = "Please transcribe the following audio clip."
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt_text,
                types.Part.from_bytes(
                    data=ogg_bytes,
                    mime_type="audio/ogg",
                )
            ]
        )
        print(response.text)
    except Exception as e:
        print(f"Error during transcription: {e}")

# Global buffers for unprocessed audio and new audio count
global_buffer = np.array([], dtype=np.float32)
global_new_audio_count = 0

def audio_callback(indata, frames, time, status):
    """
    Continuously accumulate audio data into global_buffer.
    Once we exceed CHUNK_DURATION, run VAD over the entire buffer,
    transcribe speech segments, and trim out processed audio.
    """
    global global_buffer, global_new_audio_count
    if status:
        print(status, file=sys.stderr)

    # indata is shape (frames, CHANNELS). For mono, reduce to (frames,)
    new_audio = indata[:, 0]
    global_buffer = np.concatenate((global_buffer, new_audio))
    global_new_audio_count += len(new_audio)

    if global_new_audio_count < CHUNK_DURATION * SAMPLE_RATE:
        return

    # Reset counter after accumulating CHUNK_DURATION seconds of audio.
    global_new_audio_count = 0
    print("Running VAD on buffered audio...")

    # VAD expects float PCM in [-1, 1] (global_buffer is already in that range).
    speech_segments = get_speech_timestamps(
        global_buffer,
        model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True
    )
    print(f"Detected {len(speech_segments)} speech segments.")

    if len(speech_segments) < 1:
        # Clear the buffer if no speech is detected.
        global_buffer = global_buffer[0:0]
        return
    elif len(speech_segments) == 1:
        # Retain unprocessed audio starting from the beginning of the only segment.
        trim_sample = int(speech_segments[0]['start'] * SAMPLE_RATE)
        global_buffer = global_buffer[trim_sample:]
        return
    else:
        # Process from the start of the first segment to the end of the second-last segment.
        chunk_start = int(speech_segments[0]['start'] * SAMPLE_RATE)
        chunk_end = int(speech_segments[-2]['end'] * SAMPLE_RATE)
        chunk = global_buffer[chunk_start:chunk_end]
        # Convert float data to int16.
        chunk_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
        process_audio_chunk(chunk_int16)
        # Retain audio starting from the end of the processed chunk.
        global_buffer = global_buffer[chunk_end:]

try:
    with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, callback=audio_callback):
        print("#" * 80)
        print("Recording... Press Ctrl+C to stop.")
        print("#" * 80)
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("\nRecording stopped (Ctrl+C pressed)")
except Exception as e:
    print(f"Error during audio streaming: {e}")
