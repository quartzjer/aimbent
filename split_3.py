import librosa
import soundfile as sf
import numpy as np
import time
import sys
import os

# Define functions (same as before)
def detect_vad(audio_chunk, vad_threshold=-30):
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms > vad_threshold

def detect_speaker_change(mfcc1, mfcc2, speaker_change_threshold=0.5):
    distance = np.linalg.norm(mfcc1 - mfcc2)
    return distance > speaker_change_threshold

# Main script
def main():
    if len(sys.argv) != 2:
        print("Usage: python split_test.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        sys.exit(1)

    try:
        audio, sample_rate = sf.read(audio_file)
    except sf.SoundFileError as e:
        print(f"Error reading audio file: {e}")
        sys.exit(1)

    start_time = time.time()
    print("Timestamp (seconds)") #header
    chunk_size = 1024
    mfccs_prev = None
    is_prev_active = False

    for i in range(0, len(audio), chunk_size):
        audio_chunk = audio[i:i + chunk_size]

        is_active = detect_vad(audio_chunk)

        if i > 0 and is_active:
            if mfccs_prev is None:
                mfccs_prev = librosa.feature.mfcc(y=audio[i-chunk_size:i], sr=sample_rate)
            mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sample_rate)
            if is_prev_active and detect_speaker_change(mfccs_prev, mfccs):
                timestamp = (time.time() - start_time) + (i / sample_rate)
                print(f"{timestamp:.2f}")
            mfccs_prev = mfccs
            
        is_prev_active = is_active
        
if __name__ == "__main__":
    main()