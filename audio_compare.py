import soundcard as sc
import soundfile as sf
import threading
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AudioSegment:
    device_name: str
    start_time: float
    end_time: float
    is_louder: bool

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

def analyze_levels(audio1: np.ndarray, audio2: np.ndarray, sample_rate: int, 
                  window_ms: int = 100) -> List[AudioSegment]:
    window_size = int(sample_rate * (window_ms / 1000))
    segments = []
    
    # Calculate RMS levels for each window
    times = np.arange(0, len(audio1)) / sample_rate
    num_windows = len(audio1) // window_size
    
    current_louder = None
    segment_start = 0
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        
        rms1 = np.sqrt(np.mean(audio1[start_idx:end_idx] ** 2))
        rms2 = np.sqrt(np.mean(audio2[start_idx:end_idx] ** 2))
        
        is_1_louder = rms1 > rms2
        
        if current_louder != is_1_louder and current_louder is not None:
            segments.append(AudioSegment(
                device_name="Source 1" if current_louder else "Source 2",
                start_time=times[segment_start],
                end_time=times[start_idx],
                is_louder=current_louder
            ))
            segment_start = start_idx
        
        current_louder = is_1_louder
    
    # Add final segment
    if segment_start < len(times):
        segments.append(AudioSegment(
            device_name="Source 1" if current_louder else "Source 2",
            start_time=times[segment_start],
            end_time=times[-1],
            is_louder=current_louder
        ))
    
    return segments

def visualize_audio(audio1: np.ndarray, audio2: np.ndarray, segments: List[AudioSegment], 
                   sample_rate: int):
    times = np.arange(0, len(audio1)) / sample_rate
    
    plt.figure(figsize=(15, 8))
    
    # Plot waveforms
    plt.subplot(2, 1, 1)
    plt.plot(times, audio1, label='Source 1', alpha=0.7)
    plt.plot(times, audio2, label='Source 2', alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.title('Audio Waveforms Comparison')
    plt.ylabel('Amplitude')
    
    # Plot segments
    plt.subplot(2, 1, 2)
    for segment in segments:
        color = 'blue' if segment.device_name == "Source 1" else 'red'
        plt.axvspan(segment.start_time, segment.end_time, 
                   alpha=0.3, color=color, 
                   label=segment.device_name)
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True)
    plt.title('Louder Source Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Source')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def merge_audio_segments(audio1: np.ndarray, audio2: np.ndarray, 
                        segments: List[AudioSegment], sample_rate: int) -> np.ndarray:
    """Merge audio segments based on which source was louder."""
    merged = np.zeros_like(audio1)
    
    for segment in segments:
        start_idx = int(segment.start_time * sample_rate)
        end_idx = int(segment.end_time * sample_rate)
        source_audio = audio1 if segment.device_name == "Source 1" else audio2
        merged[start_idx:end_idx] = source_audio[start_idx:end_idx]
    
    return merged

def main():
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

    # Analyze the recordings
    segments = analyze_levels(recordings[0], recordings[1], sample_rate)
    
    # Merge the audio segments
    merged_audio = merge_audio_segments(recordings[0], recordings[1], segments, sample_rate)
    
    # Save the merged audio
    sf.write('test_compare.ogg', merged_audio, sample_rate, format='OGG')
    print("\nMerged audio saved as 'test_compare.ogg'")
    
    # Print results
    print("\nAudio Level Analysis Results:")
    print("-" * 50)
    for segment in segments:
        print(f"{segment.device_name} was louder from "
              f"{segment.start_time:.2f}s to {segment.end_time:.2f}s "
              f"({(segment.end_time - segment.start_time):.2f}s duration)")
    
    # Visualize the results
    visualize_audio(recordings[0], recordings[1], segments, sample_rate)

if __name__ == "__main__":
    main()
