import sounddevice as sd
import numpy as np
import queue
import argparse
import av

class AudioProcessor:
    def __init__(self):
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()
        
        # Get default devices
        devices = sd.query_devices()
        self.mic_device = None
        self.system_device = 9
        
        # Find PulseAudio monitor device for system audio
        for i, device in enumerate(devices):
            print(f"Device {i}: {device}")
            if 'Monitor' in device['name']:
                self.system_device = i
            elif device['max_input_channels'] > 0:
                self.mic_device = i
        print(f"Selected devices - Mic: {self.mic_device}, System: {self.system_device}")

    def mic_callback(self, indata, frames, time, status):
        """Callback for microphone recording"""
        if status:
            print(f'Mic status: {status}')
        self.mic_queue.put(indata.copy())
    
    def system_callback(self, indata, frames, time, status):
        """Callback for system audio recording"""
        if status:
            print(f'System status: {status}')
        self.system_queue.put(indata.copy())
    
    def process_audio(self, duration=10, samplerate=44100):
        """Record and process audio from both sources"""
        
        # Start recording streams
        with sd.InputStream(device=self.mic_device,
                          channels=1,
                          callback=self.mic_callback,
                          samplerate=samplerate):
            with sd.InputStream(device=self.system_device,
                              channels=2,  # System audio is usually stereo
                              callback=self.system_callback,
                              samplerate=samplerate):
                
                sd.sleep(int(duration * 1000))  # Convert to milliseconds
        
        # Process recorded audio
        mic_data = []
        system_data = []
        
        # Get all data from queues
        while not self.mic_queue.empty():
            mic_data.append(self.mic_queue.get())
        while not self.system_queue.empty():
            system_data.append(self.system_queue.get())
        
        # Convert to numpy arrays
        mic_data = np.concatenate(mic_data) if mic_data else np.array([])
        system_data = np.concatenate(system_data) if system_data else np.array([])
        
        return mic_data, system_data

# Example usage
def main():
    parser = argparse.ArgumentParser(description="Record audio and save to a .m4a file.")
    parser.add_argument("output", help="Output path for the .m4a file")
    args = parser.parse_args()

    processor = AudioProcessor()
    print("Recording for 5 seconds...")
    mic_audio, system_audio = processor.process_audio(duration=5)

    # Convert float32 audio (range -1.0 to 1.0) to int16
    if mic_audio.size:
        mic_audio = (np.clip(mic_audio, -1, 1) * 32767).astype(np.int16)
    if system_audio.size:
        system_audio = (np.clip(system_audio, -1, 1) * 32767).astype(np.int16)
    
    # Save the recorded audio to the specified .m4a file with separate tracks
    samplerate = 44100

    # Open an output container for the given file path in 'ipod' format.
    output_container = av.open(args.output, mode='w', format='ipod')

    # Add two audio streams (AAC encoded)
    mic_stream = output_container.add_stream('aac', rate=samplerate)
    mic_stream.options = {'profile': 'aac_low'}

    system_stream = output_container.add_stream('aac', rate=samplerate)
    system_stream.options = {'profile': 'aac_low'}

    # Convert the NumPy arrays into AudioFrame objects.
    # Note: Using all data in a single frame.
    mic_frame = av.AudioFrame.from_ndarray(
        mic_audio.reshape(1, -1),  # Reshape to [channels, samples]
        format='s16', 
        layout='mono'
    )
    mic_frame.sample_rate = samplerate

    system_frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(system_audio.T),
        format='s16p', 
        layout='stereo'
    )
    system_frame.sample_rate = samplerate

    # Encode the frames and mux the packets into the container.
    for packet in mic_stream.encode(mic_frame):
        output_container.mux(packet)
    for packet in system_stream.encode(system_frame):
        output_container.mux(packet)

    # Flush encoders:
    for packet in mic_stream.encode():
        output_container.mux(packet)
    for packet in system_stream.encode():
        output_container.mux(packet)

    # Finalize and close the container.
    output_container.close()
    print(f"Created {args.output}")

if __name__ == "__main__":
    main()