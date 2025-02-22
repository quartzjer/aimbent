import soundcard as sc
import numpy as np
import soundfile as sf
from webrtc_audio import WebRTCAudio, WebRTCAudioError
import threading  # new import

# Initialize the WebRTC AudioProcessing module
audio = WebRTCAudio()

print("Creating APM instance...")
apm = audio.create_apm()
print(f"APM instance created: {apm != audio.ffi.NULL}")

print("Configuring APM...")
audio.configure_apm(apm)
print("APM configured")

# Open the two audio sources
print("Setting up audio devices...")
mics = sc.all_microphones(include_loopback=True)
if len(mics) < 2:
    raise RuntimeError("Need at least two microphones for echo cancellation")
far_mic = mics[1]
near_mic = mics[0]
SAMPLE_RATE = 48000
FRAME_SIZE = 480  # Exactly 10ms at 48kHz
NUM_FRAMES = int((SAMPLE_RATE * 5) / FRAME_SIZE)
channels = 1  # We'll convert stereo to mono

print(f"Using far-end reference mic: {far_mic.name}")
print(f"Using near-end mic: {near_mic.name}")

far_frames = []
near_frames = []

def record_far():
    with far_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as far_rec:
        # Record 5 seconds in one go
        far_data = far_rec.record(numframes=SAMPLE_RATE * 5)
        far_frames.append(far_data)
        print("Far mic: recorded 5 seconds")

def record_near():
    with near_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as near_rec:
        # Record 5 seconds in one go
        near_data = near_rec.record(numframes=SAMPLE_RATE * 5)
        near_frames.append(near_data)
        print("Near mic: recorded 5 seconds")

try:
    print("Starting concurrent audio capture threads...")
    t_far = threading.Thread(target=record_far)
    t_near = threading.Thread(target=record_near)
    t_far.start()
    t_near.start()
    t_far.join()
    t_near.join()
    print("Finished recording. Starting processing...")

    far_data = far_frames[0]
    near_data = near_frames[0]
    # Save raw recordings
    print("Saving raw audio recordings...")
    sf.write('test_far.ogg', far_data, SAMPLE_RATE)
    sf.write('test_near.ogg', near_data, SAMPLE_RATE)

    # Split the full recordings into chunks of FRAME_SIZE samples
    processed_frames = []
    far_data = far_frames[0]
    near_data = near_frames[0]
    num_chunks = len(far_data) // FRAME_SIZE

    for i in range(num_chunks):
        start = i * FRAME_SIZE
        end = start + FRAME_SIZE
        far_chunk = np.ascontiguousarray(far_data[start:end, 0], dtype=np.float32)
        near_chunk = np.ascontiguousarray(near_data[start:end, 0], dtype=np.float32)
        try:
            processed_audio = audio.process_frame(apm, far_chunk, near_chunk)
            processed_frames.append(processed_audio)
            if i % 50 == 0:
                print(f"Processing: chunk {i}/{num_chunks}, range: [{processed_audio.min():.3f}, {processed_audio.max():.3f}]")
        except WebRTCAudioError as e:
            print(f"Chunk {i} processing error: {e}")
            continue

    print("Processing complete!")
    print("Saving processed audio to test_aec.ogg...")
    final_audio = np.concatenate(processed_frames)
    sf.write('test_aec.ogg', final_audio, SAMPLE_RATE)
    print("Audio saved successfully!")

except Exception as e:
    print(f"Error during processing: {str(e)}")
    raise
finally:
    print("Cleaning up APM...")
    if apm != audio.ffi.NULL:
        audio.destroy_apm(apm)
        print("APM destroyed")
    print("Finished echo cancellation processing.")