import soundcard as sc
import numpy as np
from webrtc_audio import WebRTCAudio

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
mics = sc.all_microphones()
if len(mics) < 2:
    raise RuntimeError("Need at least two microphones for echo cancellation")
far_mic = mics[1]
near_mic = mics[0]
SAMPLE_RATE = 48000
FRAME_SIZE = 480  # Exactly 10ms at 48kHz
channels = 1  # We'll convert stereo to mono

print(f"Using far-end reference mic: {far_mic.name}")
print(f"Using near-end mic: {near_mic.name}")

try:
    print("Opening audio streams...")
    with far_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as far_rec, \
         near_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as near_rec:
        
        # Add debugging for initial frame capture
        print("Testing initial frame capture...")
        far_test = far_rec.record(numframes=FRAME_SIZE)
        near_test = near_rec.record(numframes=FRAME_SIZE)
        print(f"Initial far frame shape: {far_test.shape}, dtype: {far_test.dtype}")
        print(f"Initial near frame shape: {near_test.shape}, dtype: {near_test.dtype}")
        
        print("Starting processing loop...")
        NUM_FRAMES = 10  # Start with fewer frames for testing
        for i in range(NUM_FRAMES):
            print(f"\nProcessing frame {i}...")
            
            # Debug frame capture
            far_frame = far_rec.record(numframes=FRAME_SIZE)
            near_frame = near_rec.record(numframes=FRAME_SIZE)
            print(f"Captured frame shapes - far: {far_frame.shape}, near: {near_frame.shape}")
            
            # Debug mono conversion
            far_mono = np.ascontiguousarray(far_frame[:, 0], dtype=np.float32)
            near_mono = np.ascontiguousarray(near_frame[:, 0], dtype=np.float32)
            out_buf = np.zeros(FRAME_SIZE, dtype=np.float32)
            print(f"Mono arrays - far: {far_mono.shape}, near: {near_mono.shape}")
            
            # Keep arrays alive during processing
            arrays = [far_mono, near_mono, out_buf]
            
            # Debug array properties
            print(f"far_mono contiguous: {far_mono.flags['C_CONTIGUOUS']}")
            print(f"near_mono contiguous: {near_mono.flags['C_CONTIGUOUS']}")
            print(f"Data ranges - far: [{far_mono.min():.3f}, {far_mono.max():.3f}], "
                  f"near: [{near_mono.min():.3f}, {near_mono.max():.3f}]")
            
            # Debug processing
            print("Calling process_frame through WebRTCAudio API...")
            err = audio.process_frame(apm, far_mono, near_mono, out_buf,
                                      SAMPLE_RATE, channels, FRAME_SIZE)
            if err != 0:
                error_messages = {
                    -1: "Invalid parameters",
                    -2: "Processing exception",
                    -3: "Invalid sample rate",
                    -4: "Invalid channel count",
                    -5: "Invalid frame size"
                }
                print(f"Error: {error_messages.get(err, 'Unknown error')} (code: {err})")
                continue
            
            print(f"Output range: [{out_buf.min():.3f}, {out_buf.max():.3f}]")
            
except Exception as e:
    print(f"Error during processing: {str(e)}")
    raise
finally:
    print("Cleaning up APM...")
    if apm != audio.ffi.NULL:
        audio.destroy_apm(apm)
        print("APM destroyed")
    print("Finished echo cancellation processing.")