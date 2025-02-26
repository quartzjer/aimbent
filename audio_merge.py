import soundcard as sc
import soundfile as sf
import threading
import numpy as np
import argparse
from typing import Tuple

def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono by averaging channels."""
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        return np.mean(audio, axis=1)
    return audio

def record_device(mic, sample_rate, duration, device_id) -> np.ndarray:
    print(f"Recording {duration} seconds using device {device_id}: {mic.name}")
    try:
        recording = mic.record(samplerate=sample_rate, numframes=int(duration * sample_rate))
        return to_mono(recording)
    except Exception as e:
        print(f"Error recording from device {device_id}: {e}")
        return np.array([])

def merge_with_latency(audio1: np.ndarray, audio2: np.ndarray, 
                      latency_ms: float, sample_rate: int) -> np.ndarray:
    """Merge two audio streams with the second stream shifted by latency_ms."""
    # Convert latency from ms to samples
    shift_samples = int((latency_ms / 1000) * sample_rate)
    
    # Create shifted version of audio2
    shifted_audio2 = np.zeros_like(audio1)
    if shift_samples >= 0:
        shifted_audio2[shift_samples:] = audio2[:-shift_samples] if shift_samples > 0 else audio2
    else:
        shifted_audio2[:shift_samples] = audio2[-shift_samples:]
    
    # Combine both streams (average them)
    merged = (audio1 + shifted_audio2) / 2
    return merged

def main():
    parser = argparse.ArgumentParser(description='Record and merge audio with latency compensation')
    parser.add_argument('--latency', type=float, default=0,
                       help='Latency in milliseconds to shift second source forward')
    args = parser.parse_args()

    mics = sc.all_microphones(include_loopback=True)
    if len(mics) < 2:
        print("Need at least 2 audio devices!")
        return

    print("Using first two available audio devices:")
    for i in range(2):
        print(f"{i}: {mics[i].name}")

    sample_rate = 16000
    duration = 5

    # Record both sources in parallel
    recordings = [None, None]
    threads = []
    
    for i in range(2):
        thread = threading.Thread(
            target=lambda idx: recordings.__setitem__(idx, 
                record_device(mics[idx], sample_rate, duration, idx)),
            args=(i,)
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Merge the audio with latency compensation
    merged_audio = merge_with_latency(recordings[0], recordings[1], args.latency, sample_rate)
    
    # Save the merged audio
    sf.write('test_compare.ogg', merged_audio, sample_rate, format='OGG')
    print(f"\nMerged audio saved as 'test_compare.ogg' with {args.latency}ms latency compensation")

if __name__ == "__main__":
    main()
