import os
import sys
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from dotenv import load_dotenv
from google import genai
from google.genai import types
import subprocess

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit("Error: GOOGLE_API_KEY not found in environment. Please set it in your .env file.")

if len(sys.argv) < 3:
    sys.exit("Usage: python gemini_audio.py <chunk_duration_sec> <path_to_audio_or_video_file>")

chunk_duration_sec = float(sys.argv[1])
file_path = sys.argv[2]
if not os.path.isfile(file_path):
    sys.exit(f"Error: File '{file_path}' not found.")

# Load and analyze the audio
model = load_silero_vad()
wav = read_audio(file_path, sampling_rate=16000)
speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)

# Get file duration using ffprobe
result = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=noprint_wrappers=1:nokey=1", file_path],
    capture_output=True, text=True
)
file_duration = float(result.stdout.strip())

chunk_target = chunk_duration_sec
chunks = []
current_time = 0.0
while current_time < file_duration:
    target_time = current_time + chunk_target
    boundary = file_duration
    for seg in speech_timestamps:
        if seg["end"] >= target_time and file_duration - seg["end"] > chunk_target:
            boundary = seg["end"]
            break
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(current_time),
        "-to", str(boundary),
        "-i", file_path,
        "-c", "copy",
        "-f", "mp3",
        "pipe:1"
    ]
    proc = subprocess.run(ffmpeg_cmd, capture_output=True)
    if proc.returncode != 0:
        print(f"Error extracting chunk from {current_time} to {boundary}")
        break
    chunks.append((proc.stdout, current_time, boundary))
    current_time = boundary

client = genai.Client(api_key=API_KEY)

# Transcribe each chunk using the inline bytes from ffmpeg
for idx, (chunk_bytes, start_time, end_time) in enumerate(chunks, start=1):
    prompt_text = f"Please transcribe the following audio clip."
    duration = end_time - start_time
    size_mb = len(chunk_bytes) / (1024 * 1024)
    print(f"\n--- Transcription for chunk {idx}: {duration:.1f}s, {size_mb:.1f}MB ---")

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt_text,
                types.Part.from_bytes(
                    data=chunk_bytes,
                    mime_type="audio/mp3",
                )
            ]
        )
        print(response.text)
    except Exception as e:
        print(f"Error during transcription of chunk {idx}: {e}")
