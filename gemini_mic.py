import os
import sys
import io
import numpy as np
import sounddevice as sd
import av
from silero_vad import load_silero_vad, get_speech_timestamps
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit("Error: GOOGLE_API_KEY not found in environment. Please set it in your .env file.")

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5.0  # seconds

# Load Silero VAD model and initialize Gemini client
model = load_silero_vad()
client = genai.Client(api_key=API_KEY)

def process_audio_chunk(audio_chunk):
    """
    Encode the given audio chunk as AAC and send it to Gemini for transcription.
    `audio_chunk` here is assumed to be int16 NumPy array.
    """
    # Convert the NumPy int16 array to AAC using PyAV
    buf = io.BytesIO()
    output_container = av.open(buf, mode='w', format='ipod')
    stream = output_container.add_stream('aac', rate=SAMPLE_RATE)
    stream.options = {'profile': 'aac_low'}

    # Create a mono audio frame. If audio_chunk is shape (N,), reshape to (1, N) for PyAV.
    frame = av.AudioFrame.from_ndarray(audio_chunk.reshape(1, -1), format='s16', layout='mono')
    frame.sample_rate = SAMPLE_RATE

    for packet in stream.encode(frame):
        output_container.mux(packet)
    for packet in stream.encode():
        output_container.mux(packet)

    output_container.close()
    aac_bytes = buf.getvalue()

    duration_sec = len(audio_chunk) / SAMPLE_RATE
    size_mb = len(aac_bytes) / (1024 * 1024)
    print(f"Transcribing chunk: {duration_sec:.1f}s, {size_mb:.2f}MB")

    prompt_text = "Please transcribe the following audio clip."
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt_text,
                types.Part.from_bytes(
                    data=aac_bytes,
                    mime_type="audio/aac",
                )
            ]
        )
        print(response.text)
    except Exception as e:
        print(f"Error during transcription: {e}")

# Global buffer to accumulate microphone data as float32 (default for sounddevice)
global_buffer = np.array([], dtype=np.float32)

def audio_callback(indata, frames, time, status):
    """
    Continuously accumulate audio data into global_buffer.
    Once we exceed CHUNK_DURATION, run VAD over the entire buffer,
    transcribe speech segments, and trim out processed audio.
    """
    global global_buffer
    if status:
        print(status, file=sys.stderr)

    # indata is shape (frames, CHANNELS). For mono, we can reduce to (frames,)
    # If multi-channel, adapt accordingly (just be consistent).
    new_audio = indata[:, 0]

    # Append to the global buffer
    global_buffer = np.concatenate((global_buffer, new_audio))

    # Check if we have at least CHUNK_DURATION seconds in the buffer
    total_length_sec = len(global_buffer) / SAMPLE_RATE
    if total_length_sec < CHUNK_DURATION:
        return  # Not enough data to process yet

    # We have enough data to do some VAD
    print("Running VAD on buffered audio...")

    # VAD expects float PCM in [-1, 1]
    # global_buffer is already float32 in [-1, 1], so no need to scale for VAD.
    speech_segments = get_speech_timestamps(
        global_buffer,
        model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True
    )
    print(f"Detected {len(speech_segments)} speech segments.")

    # If we found no speech, flush out the oldest chunk's worth
    if not speech_segments:
        # Just drop CHUNK_DURATION worth of samples to avoid ballooning the buffer
        drop_samples = int(CHUNK_DURATION * SAMPLE_RATE)
        global_buffer = global_buffer[drop_samples:]
        return

    # Otherwise, process each segment and track the max end
    last_end_sample = 0
    for seg in speech_segments:
        start_sec = seg['start']
        end_sec = seg['end']
        start_sample = int(start_sec * SAMPLE_RATE)
        end_sample = int(end_sec * SAMPLE_RATE)

        # Only process forward in the buffer (avoid reprocessing if segments overlap)
        if end_sample <= last_end_sample:
            continue

        # Clip to not reprocess previously handled frames
        segment_start = max(start_sample, last_end_sample)
        segment_end = end_sample

        chunk = global_buffer[segment_start:segment_end]
        # Convert float to int16 for PyAV:
        chunk_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)

        process_audio_chunk(chunk_int16)
        last_end_sample = segment_end

    # After processing all segments, remove everything up to last_end_sample
    global_buffer = global_buffer[last_end_sample:]

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
