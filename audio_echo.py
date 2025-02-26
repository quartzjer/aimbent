import soundcard as sc
import soundfile as sf
import threading
import numpy as np
from pyaec import Aec
import struct

def record_device(device, sample_rate, duration, storage, key):
    # This call blocks until the recording is complete.
    storage[key] = device.record(samplerate=sample_rate, numframes=int(duration * sample_rate))

def process_with_pyaec(mic_audio, monitor_audio, sample_rate):
    """
    Apply echo cancellation on mic_audio using monitor_audio as the far-end reference
    using PyAEC library.
    
    Args:
        mic_audio: Numpy array containing microphone audio (near-end)
        monitor_audio: Numpy array containing system audio (far-end)
        sample_rate: Audio sample rate in Hz
    
    Returns:
        Numpy array containing echo-cancelled audio
    """
    # PyAEC parameters
    frame_size = 160  # 10ms at 16kHz
    filter_length = int(sample_rate * 0.1)
    
    # Create echo canceller
    aec = Aec(frame_size, filter_length, sample_rate, True)
    
    # Convert float32 arrays to int16 (required by PyAEC)
    mic_int16 = (mic_audio[:,0] * 32767).astype(np.int16)
    monitor_int16 = (monitor_audio[:,0] * 32767).astype(np.int16)
    
    # Process in chunks of frame_size
    processed_chunks = []
    for i in range(0, len(mic_int16) - frame_size, frame_size):
        mic_chunk = mic_int16[i:i+frame_size]
        monitor_chunk = monitor_int16[i:i+frame_size]
        
        # Process with echo canceller
        processed_chunk = aec.cancel_echo(mic_chunk, monitor_chunk)
        processed_chunks.append(processed_chunk)
    
    # Combine chunks and convert back to float32 range [-1, 1]
    if processed_chunks:
        processed = np.concatenate(processed_chunks).astype(np.float32) / 32767.0
        return processed
    else:
        return np.array([])

def main():
    # List devices (including loopback/monitor devices)
    devices = sc.all_microphones(include_loopback=True)
    if len(devices) < 2:
        print("Need at least 2 audio devices (mic and monitor)!")
        return

    # Assume the first is the physical mic and the second is the system monitor.
    mic = devices[0]
    monitor = devices[1]
    print("Using devices:")
    print(f"Input microphone: {mic.name}")
    print(f"System monitor: {monitor.name}")
    
    sample_rate = 16000  # Use 16kHz as recommended by PyAEC example
    duration = 5         # Duration in seconds

    recordings = {}
    # Create threads to record simultaneously.
    t_mic = threading.Thread(target=record_device, args=(mic, sample_rate, duration, recordings, 'mic'))
    t_monitor = threading.Thread(target=record_device, args=(monitor, sample_rate, duration, recordings, 'monitor'))
    t_mic.start()
    t_monitor.start()
    t_mic.join()
    t_monitor.join()

    # Retrieve the concurrently recorded data.
    mic_recording = recordings.get('mic')
    monitor_recording = recordings.get('monitor')

    if mic_recording is None or monitor_recording is None:
        print("Recording failed.")
        return

    # Save original recordings
    sf.write("test_mic.ogg", mic_recording, sample_rate)
    sf.write("test_system.ogg", monitor_recording, sample_rate)
    print("Original recordings saved to test_mic.ogg and test_system.ogg")

    # Process the mic audio using the monitor audio as reference.
    processed = process_with_pyaec(mic_recording, monitor_recording, sample_rate)

    if len(processed) > 0:
        # Save the processed (echo-cancelled) audio.
        output_filename = "processed_echo_cancelled.ogg"
        # Ensure output is two-dimensional (mono)
        sf.write(output_filename, processed.reshape(-1, 1), sample_rate)
        print(f"Processed recording saved to {output_filename}")
    else:
        print("Error: No processed audio was generated.")

if __name__ == "__main__":
    main()
