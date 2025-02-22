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
        print("No audio devices found!")
        return

    print("Available audio devices:")
    for i, mic in enumerate(mics):
        print(f"{i}: {mic.name}")
    
    # Get user input for device selection
    while True:
        try:
            selection = input("\nEnter device IDs to record from (comma-separated): ").strip()
            device_ids = [int(x.strip()) for x in selection.split(',')]
            if not all(0 <= id < len(mics) for id in device_ids):
                raise ValueError("Invalid device ID")
            break
        except ValueError:
            print(f"Please enter valid device IDs between 0 and {len(mics)-1}")
    
    sample_rate = 44100
    duration = 5

    print("\nRecording from selected devices in parallel...")
    threads = []
    for device_id in device_ids:
        thread = threading.Thread(target=record_device, args=(mics[device_id], sample_rate, duration, device_id))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
