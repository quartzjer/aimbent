import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import threading

def play_vocal_test_tones(duration=3.0, sample_rate=44100):
    """
    Generates and plays a series of tones in the vocal range to help stimulate
    echo in the environment.
    
    Args:
        duration (float): Duration of the test in seconds
        sample_rate (int): Audio sample rate in Hz
    """
    print("Playing test tones to stimulate echo...")
    
    # Common frequencies in vocal range (Hz)
    frequencies = [150, 300, 1000]
    
    # Generate time array
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Generate signal with different frequencies
    signal = np.zeros_like(t)
    segment_duration = duration / len(frequencies)
    
    for i, freq in enumerate(frequencies):
        start_idx = int(i * segment_duration * sample_rate)
        end_idx = int((i + 1) * segment_duration * sample_rate)
        segment = 0.15 * np.sin(2 * np.pi * freq * t[start_idx:end_idx])
        signal[start_idx:end_idx] = segment
    
    # Apply fade in/out
    fade_samples = int(0.1 * sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out
    
    # Play the signal
    sd.play(signal, sample_rate)
    sd.wait()  # Wait until sound has finished playing

def detect_echo_threshold(duration=5.0, sample_rate=44100, percentile=95.0, use_test_tones=True):
    """
    Records audio from the default microphone for the specified duration
    and calculates a threshold that will gate the specified percentile
    of the input signal amplitudes. Optionally plays test tones first to help
    stimulate echo.
    
    Args:
        duration (float): Recording duration in seconds
        sample_rate (int): Audio sample rate in Hz
        percentile (float): Percentile of input to gate (0-100)
        use_test_tones (bool): Whether to play test tones before recording
        
    Returns:
        float: The calculated threshold level
    """
    if use_test_tones:
        print("Playing test tones to stimulate echo...")
        tone_thread = threading.Thread(target=play_vocal_test_tones, args=(3.0, sample_rate))
        tone_thread.start()
        print("Now recording the echo response concurrently...")
    else:
        print(f"Recording {duration} seconds of audio to detect echo threshold...")
        print("Please ensure only echo is present during recording...")
    
    # Record audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, 
                      channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    
    if use_test_tones:
        tone_thread.join()  # Wait for the tone thread to finish
    
    # Flatten the array and get absolute amplitude values
    audio_data = recording.flatten()
    amplitudes = np.abs(audio_data)
    
    # Calculate the threshold at the specified percentile
    threshold = np.percentile(amplitudes, percentile)
    
    print(f"Analysis complete: {len(audio_data)} samples analyzed")
    print(f"Detected threshold: {threshold:.6f} (gates {percentile}% of input)")
    
    return threshold

def visualize_threshold(audio_data, threshold):
    """
    Creates a visualization of the audio data and the threshold level.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(audio_data)
    plt.axhline(y=threshold, color='r', linestyle='-', label=f'Threshold ({threshold:.6f})')
    plt.axhline(y=-threshold, color='r', linestyle='-')
    plt.ylabel('Amplitude')
    plt.xlabel('Sample')
    plt.title('Audio Waveform with Gate Threshold')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage
    duration = 5.0  # seconds
    sample_rate = 16000
    threshold = detect_echo_threshold(duration=duration, sample_rate=sample_rate, percentile=98, use_test_tones=False)
    print(f"Recommended gate threshold: {threshold:.6f}")
    
    # Record user speech for visualization
    input("Press Enter and start speaking to record a 5-second sample for visualization...")
    print("Recording your speech for visualization...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    visualize_threshold(audio.flatten(), threshold)
