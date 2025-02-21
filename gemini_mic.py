import os
import sys
import io
import numpy as np
import sounddevice as sd
import soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps
from dotenv import load_dotenv
from google import genai
from google.genai import types
import threading
import datetime
import json
from noisereduce import reduce_noise

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

# Load Silero VAD model and initialize Gemini client
model = load_silero_vad()
client = genai.Client(api_key=API_KEY)

global_speech_segments = []  # (float32 arrays)
speech_segments_lock = threading.Lock()  # lock for thread-safe operations on global_speech_segments

def process_audio_chunk(ogg_bytes):
    """
    Send the already encoded ogg_bytes to Gemini for transcription.
    Returns the response text.
    """
    size_mb = len(ogg_bytes) / (1024 * 1024)
    print(f"Transcribing chunk: {size_mb:.2f}MB")
    # Load prompt from file
    with open("gemini_mic.txt", "r") as f:
        prompt_text = f.read().strip()
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            #model="gemini-2.0-pro-exp-02-05",
            contents=[
                "Process the provided audio now and output your professional accurate transcription in the specified JSON format.",
                types.Part.from_bytes(
                    data=ogg_bytes,
                    mime_type="audio/ogg",
                )
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

# for unprocessed audio and counter for new audio
global_buffer = np.array([], dtype=np.float32)
global_new_audio_count = 0
buffer_lock = threading.Lock()

def process_buffer(buffer_snapshot):
    """
    Run VAD on the provided buffer snapshot.
    Process each detected segment; if a segment's end is less than 1s from the buffer's end,
    stop processing further segments and return that fragment as unprocessed.
    """
    total_duration = len(buffer_snapshot) / SAMPLE_RATE
    speech_segments = get_speech_timestamps(
        buffer_snapshot,
        model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        speech_pad_ms=100,
        min_silence_duration_ms=500
    )
    print(f"Detected {len(speech_segments)} speech segments.")

    if not speech_segments:
        return None

    complete_segments = []
    unprocessed = None

    for segment in speech_segments:
        # The trailing segment may be incomplete yet, leave it for next round
        if total_duration - segment['end'] < 1:
            unprocessed = buffer_snapshot[int(segment['start'] * SAMPLE_RATE):]
            break
        start_idx = int(segment['start'] * SAMPLE_RATE)
        end_idx = int(segment['end'] * SAMPLE_RATE)
        complete_segments.append(buffer_snapshot[start_idx:end_idx])
    
    if complete_segments:
        with speech_segments_lock:
            global_speech_segments.extend(complete_segments)
    
    print(f"Keeping {len(complete_segments)} complete segments, returning {len(unprocessed) / SAMPLE_RATE if unprocessed is not None else 0}s as unprocessed.")
    return unprocessed

def combine_speech_chunks_timer():
    """
    Every 60 seconds combine collected speech segments, create an OGG, transcribe it,
    and save the OGG and transcription response to timestamped files.
    """
    while True:
        sd.sleep(60000)
        with speech_segments_lock:
            if not global_speech_segments:
                continue
            # silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
            # combined = global_speech_segments[0]
            # for seg in global_speech_segments[1:]:
            #     combined = np.concatenate((combined, silence, seg))
            combined = np.concatenate(global_speech_segments)
            count = len(global_speech_segments)
            global_speech_segments.clear()
        
        # Insert noise reduction before converting:
        combined_clean = reduce_noise(y=combined, sr=SAMPLE_RATE)
        chunk_int16 = (np.clip(combined_clean, -1.0, 1.0) * 32767).astype(np.int16)
        
        # Create OGG bytes from combined chunks.
        buf = io.BytesIO()
        audio_data = chunk_int16.reshape(-1, CHANNELS)
        sf.write(buf, audio_data, SAMPLE_RATE, format='OGG', subtype='VORBIS')
        ogg_bytes = buf.getvalue()
        
        print(f"Transcribing {count} combined speech segments.")
        response_text = process_audio_chunk(ogg_bytes)
        
        # Get timestamp and save files.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ogg_filepath = os.path.join(save_dir, f"audio_{timestamp}.ogg")
        json_filepath = os.path.join(save_dir, f"audio_{timestamp}.json")
        with open(ogg_filepath, "wb") as f:
            f.write(ogg_bytes)
        with open(json_filepath, "w") as f:
            json.dump({"text": response_text}, f)
        print(f"Saved to {ogg_filepath}: {response_text}")

def process_buffer_async(buffer_snapshot):
    """
    Execute VAD processing in a background thread and merge unprocessed audio back.
    """
    unprocessed = process_buffer(buffer_snapshot)
    if unprocessed is not None:
        with buffer_lock:
            global global_buffer
            global_buffer = np.concatenate((unprocessed, global_buffer))

def audio_callback(indata, frames, time, status):
    """
    Continuously accumulate audio data into global_buffer.
    When enough audio is accumulated, offload VAD and transcription processing to a background thread.
    """
    global global_buffer, global_new_audio_count
    if status:
        print(status, file=sys.stderr)

    # indata is shape (frames, CHANNELS). For mono, reduce to (frames,)
    new_audio = indata[:, 0]
    with buffer_lock:
        global_buffer = np.concatenate((global_buffer, new_audio))
        global_new_audio_count += len(new_audio)
        if global_new_audio_count < CHUNK_DURATION * SAMPLE_RATE:
            return
        # Take a snapshot and reset the buffer and counter.
        buffer_snapshot = global_buffer.copy()
        global_buffer = np.array([], dtype=np.float32)
        global_new_audio_count = 0

    print("Starting background VAD processing...")
    threading.Thread(target=process_buffer_async, args=(buffer_snapshot,)).start()

# Start the background timer thread.
timer_thread = threading.Thread(target=combine_speech_chunks_timer, daemon=True)
timer_thread.start()

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
