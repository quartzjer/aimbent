import soundcard as sc
import soundfile as sf
import threading

def record_device(mic, sample_rate, duration, device_id):
    # Record audio from a single device and save to 'test_[id].ogg'
    print(f"\nRecording {duration} seconds using device {device_id}: {mic.name}")
    try:
        recording = mic.record(samplerate=sample_rate, numframes=int(duration * sample_rate))
        filename = f"test_{device_id}.ogg"
        sf.write(filename, recording, sample_rate)
        print(f"Recording from device {device_id} saved to {filename}")
    except Exception as e:
        print(f"Error recording from device {device_id}: {e}")

def main():
    # List all microphones including loopback (monitor) devices.
    mics = sc.all_microphones(include_loopback=True)
    if not mics:
        print("No audio input devices found!")
        return

    print("Available audio input devices (including loopback devices):")
    for i, mic in enumerate(mics):
        # mic.channels is an integer (number of channels)
        print(f"{i}: {mic.name} (channels: {mic.channels})")
    
    sample_rate = 44100  # or 48000, depending on your system's settings
    duration = 5         # seconds

    print("\nRecording from all devices in parallel...")
    threads = []
    for i, mic in enumerate(mics):
        thread = threading.Thread(target=record_device, args=(mic, sample_rate, duration, i))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
