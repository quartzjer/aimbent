import soundcard as sc
import soundfile as sf

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
    
    try:
        device_id = int(input("\nEnter device ID for recording: "))
        if device_id < 0 or device_id >= len(mics):
            print(f"Error: Device ID must be between 0 and {len(mics)-1}")
            return
        
        mic = mics[device_id]
        sample_rate = 44100  # or 48000, depending on your system's settings
        duration = 5         # seconds
        
        print(f"\nRecording {duration} seconds using device: {mic.name}")
        print("Recording...")
        
        # Record audio (numframes = sample_rate * duration)
        recording = mic.record(samplerate=sample_rate, numframes=int(duration * sample_rate))
        
        # Save the recording to a file
        sf.write('test.ogg', recording, sample_rate)
        print("Recording saved to test.ogg")
        
    except ValueError:
        print("Error: Please enter a valid integer.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
