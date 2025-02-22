import soundcard as sc
import soundfile as sf
import threading
import numpy as np
import adaptfilt as adf  # Using adaptfilt for adaptive filtering

def record_device(device, sample_rate, duration, storage, key):
    # This call blocks until the recording is complete.
    storage[key] = device.record(samplerate=sample_rate, numframes=int(duration * sample_rate))

def process_with_echo_cancellation(mic_audio, monitor_audio, sample_rate):
    """
    Apply echo cancellation on mic_audio using monitor_audio as the far-end reference.
    Assumes mic_audio and monitor_audio are numpy arrays in float32 format (range -1 to 1).
    
    This implementation uses the NLMS adaptive filter from adaptfilt:
      - x: reference (monitor) signal.
      - d: microphone signal (contains near-end speech plus echo).
      - The NLMS algorithm estimates the echo component so that the error e = d - y
        contains the near-end speech with the echo removed.
    """
    # Parameters for the NLMS algorithm (tune as necessary)
    mu = 0.005        # Step size
    filter_order = 1024  # Number of filter taps (must be an integer)
    
    # Flatten signals to ensure they are 1D arrays.
    x = monitor_audio[:,0].flatten()  # Far-end (reference) signal.
    d = mic_audio[:,0].flatten()      # Mic signal (with echo).

    # NOTE: The correct call is: nlms(u, d, M, step, ...)
    # Swap the order: third argument must be filter_order, fourth is mu.
#    y, e, w = adf.nlms(x, d, filter_order, mu)

    filter_order = 512
    lam = 0.99  # forgetting factor
    delta = 0.001
    
    y, e, w = adf.rls(x, d, filter_order, lam, delta)
    return e

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
    
    sample_rate = 44100  # Samples per second
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
    processed = process_with_echo_cancellation(mic_recording, monitor_recording, sample_rate)

    # Save the processed (echo-cancelled) audio.
    output_filename = "processed_echo_cancelled.ogg"
    # Ensure output is two-dimensional (mono)
    sf.write(output_filename, processed.reshape(-1, 1), sample_rate)
    print(f"Processed recording saved to {output_filename}")

if __name__ == "__main__":
    main()
