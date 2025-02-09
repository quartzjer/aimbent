#!/usr/bin/env python3
"""
split_test.py

This script demonstrates how to:
1. Load an MP3 file.
2. Perform voice activity detection (VAD) using webrtcvad.
3. Extract speaker embeddings (using Resemblyzer) from voice segments.
4. Detect speaker changes by comparing consecutive embeddings.
5. Log the timestamps in seconds where speech starts, ends, and where speaker changes occur.

Dependencies (install via pip):
- pydub        (for audio decoding) 
- webrtcvad    (for voice activity detection)
- resemblyzer  (for speaker embeddings)
- soundfile    (for writing/reading raw audio if needed)
- numpy

Usage:
    python3 split_test.py input_file.mp3

This is a minimal example and may need refinement (threshold tuning, chunk sizes, etc.).
"""

import sys
import io
import math
import numpy as np
from pydub import AudioSegment
import webrtcvad
from resemblyzer import VoiceEncoder, preprocess_wav

# ---------------- Configuration Parameters ----------------
FRAME_DURATION_MS = 30        # Duration of each frame for VAD
SAMPLE_RATE = 16000           # Target sample rate
VAD_AGGRESSIVENESS = 2        # 0-3, higher values = more aggressive
MIN_VOICE_SEGMENT_MS = 200    # Minimum voice segment length to consider speech
SPEAKER_CHANGE_THRESHOLD = 0.7  # Lower similarity implies a speaker change. Adjust as needed.
SUB_CHUNK_DURATION = 2.0      # Duration in seconds for each sub-chunk for embedding
# ----------------------------------------------------------

def frame_generator(audio_bytes, frame_duration_ms, sample_rate):
    """Yield frames of audio_bytes at frame_duration_ms intervals."""
    frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit = 2 bytes per sample
    for i in range(0, len(audio_bytes), frame_size):
        yield audio_bytes[i:i+frame_size]

def detect_speech_segments(raw_audio, sample_rate, vad_aggressiveness):
    """Use webrtcvad to detect continuous voice segments.
    Returns a list of (start_sec, end_sec, audio_data) tuples."""
    vad = webrtcvad.Vad(vad_aggressiveness)
    frames = list(frame_generator(raw_audio, FRAME_DURATION_MS, sample_rate))

    # Running boolean to keep track of when we're "in speech"
    voiced_frames = []
    segments = []
    start_time_sec = None

    frame_duration = FRAME_DURATION_MS / 1000.0
    for i, f in enumerate(frames):
        is_speech = vad.is_speech(f, sample_rate)
        t_sec = i * frame_duration
        if is_speech:
            if start_time_sec is None:
                start_time_sec = t_sec
            voiced_frames.append(f)
        else:
            if start_time_sec is not None:
                # End of a voiced segment
                end_time_sec = t_sec
                segment_duration = (end_time_sec - start_time_sec) * 1000
                if segment_duration >= MIN_VOICE_SEGMENT_MS:
                    segment_audio = b"".join(voiced_frames)
                    segments.append((start_time_sec, end_time_sec, segment_audio))
                # Reset
                start_time_sec = None
                voiced_frames = []

    # If ended with voice active
    if start_time_sec is not None and len(voiced_frames) > 0:
        end_time_sec = len(frames)*frame_duration
        segment_duration = (end_time_sec - start_time_sec)*1000
        if segment_duration >= MIN_VOICE_SEGMENT_MS:
            segment_audio = b"".join(voiced_frames)
            segments.append((start_time_sec, end_time_sec, segment_audio))

    return segments

def extract_speaker_changes(segment, start_sec, sample_rate, encoder):
    """Given a speech segment, split into sub-chunks and compute embeddings.
       Return timestamps of speaker changes relative to the entire file."""
    # Convert bytes to float wav with preprocess_wav (Resemblyzer)
    # preprocess_wav expects a numpy float32 waveform
    # We'll decode the raw 16-bit PCM here:
    import numpy as np
    import struct

    samples = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
    # Ensure single-channel at sample_rate (already should be)
    # Now we compute embeddings on sub-chunks
    chunk_size_samples = int(SUB_CHUNK_DURATION * sample_rate)
    # If segment is shorter than a single sub-chunk, just treat as one
    if len(samples) < chunk_size_samples:
        return []  # no speaker change can be reliably detected in a single short chunk

    embeddings = []
    times = []
    for i in range(0, len(samples), chunk_size_samples):
        sub_chunk = samples[i:i+chunk_size_samples]
        if len(sub_chunk) < 0.5 * chunk_size_samples:
            break  # If the last chunk is too small, skip
        emb = encoder.embed_utterance(sub_chunk)
        embeddings.append(emb)
        # Time (in sec) of this sub-chunk's start relative to the file
        sub_chunk_start_sec = start_sec + i / sample_rate
        times.append(sub_chunk_start_sec)

    # Compare consecutive embeddings
    speaker_change_times = []
    for i in range(1, len(embeddings)):
        # We'll use cosine similarity here
        emb1 = embeddings[i-1]
        emb2 = embeddings[i]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1)*np.linalg.norm(emb2))
        if similarity < SPEAKER_CHANGE_THRESHOLD:
            # Speaker change detected at boundary between these two sub-chunks
            # The boundary time is approximately times[i]
            speaker_change_times.append(times[i])
    return speaker_change_times

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 split_test.py input_file.mp3")
        sys.exit(1)

    input_file = sys.argv[1]

    # Load and resample MP3 file to mono 16k
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    raw_audio = audio.raw_data

    # Detect VAD segments
    segments = detect_speech_segments(raw_audio, SAMPLE_RATE, VAD_AGGRESSIVENESS)

    if len(segments) == 0:
        print("No speech segments detected.")
        return

    # Initialize the speaker embedding model
    encoder = VoiceEncoder()

    # We will log:
    # - Speech start and end times
    # - Speaker change times
    for (start_sec, end_sec, seg_audio) in segments:
        print(f"Speech segment: start={start_sec:.2f}s, end={end_sec:.2f}s")
        # Detect speaker changes within this segment
        speaker_changes = extract_speaker_changes(seg_audio, start_sec, SAMPLE_RATE, encoder)
        for sc in speaker_changes:
            print(f"    Speaker change detected at {sc:.2f}s")

if __name__ == "__main__":
    main()
