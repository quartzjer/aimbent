import soundcard as sc
import numpy as np
from cffi import FFI

# CFFI: Define the interface (C prototypes) for our wrapper functions
ffi = FFI()
ffi.cdef("""
    // Opaque handle for the AudioProcessing object
    typedef struct AudioProcessing AudioProcessing;
    // Create and destroy the APM (Audio Processing Module) instance
    AudioProcessing* apm_create();
    void apm_destroy(AudioProcessing* apm);
    // Configure APM (enable echo cancellation, etc.)
    void apm_configure(AudioProcessing* apm);
    // Process one 10ms frame of audio (far_end may be NULL if no reference available)
    int apm_process(AudioProcessing* apm,
                    const float* far_frame,
                    const float* near_frame,
                    float* out_frame,
                    int rate, int channels, int frame_size);
""")

# CFFI: Provide the C++ implementation of the wrapper
ffi.set_source(
    "_webrtc_apm",
    r'''
    #include <vector>
    #include <memory>
    #include <modules/audio_processing/include/audio_processing.h>
    using namespace webrtc;
    extern "C" {
        AudioProcessing* apm_create() {
            try {
                AudioProcessingBuilder builder;
                AudioProcessing::Config config;
                
                config.echo_canceller.enabled = true;
                config.echo_canceller.mobile_mode = false;
                config.echo_canceller.enforce_high_pass_filtering = true;
                config.gain_controller1.enabled = true;
                config.gain_controller1.mode = AudioProcessing::Config::GainController1::kAdaptiveDigital;
                config.noise_suppression.enabled = true;
                
                AudioProcessing* apm = builder.Create();
                if (!apm) {
                    return nullptr;
                }
                apm->ApplyConfig(config);
                return apm;
            } catch (...) {
                return nullptr;
            }
        }
        
        void apm_destroy(AudioProcessing* apm) {
            if (apm) {
                try {
                    delete apm;
                } catch (...) {
                    // Ignore any exceptions during cleanup
                }
            }
        }
        
        void apm_configure(AudioProcessing* apm) {
            if (!apm) return;
            AudioProcessing::Config config;
            config.echo_canceller.enabled = true;
            config.echo_canceller.mobile_mode = false;
            config.gain_controller1.enabled = true;
            config.gain_controller1.mode = AudioProcessing::Config::GainController1::kAdaptiveDigital;
            config.noise_suppression.enabled = true;
            apm->ApplyConfig(config);
        }
        
        int apm_process(AudioProcessing* apm,
                        const float* far_frame,
                        const float* near_frame,
                        float* out_frame,
                        int rate, int channels, int frame_size) {
            if (!apm || !near_frame || !out_frame) {
                return -1;
            }
            
            try {
                if (rate != 48000 || channels != 1 || frame_size != 480) {
                    return -3;
                }
                
                StreamConfig stream_config(rate, channels);
                
                // Process far-end (reference) audio if provided
                if (far_frame) {
                    const float* const far_channels[1] = { far_frame };
                    float far_out[480];  // Temporary buffer for far-end output
                    float* far_out_channels[1] = { far_out };  // We need a valid output buffer even if we don't use it
                    
                    int err = apm->ProcessReverseStream(far_channels, stream_config, stream_config, far_out_channels);
                    if (err != 0) {
                        return -4;
                    }
                }
                
                // Process near-end audio
                const float* const near_channels[1] = { near_frame };
                float* out_channels[1] = { out_frame };
                
                apm->set_stream_delay_ms(0);
                
                return apm->ProcessStream(near_channels, stream_config, stream_config, out_channels);
            } catch (...) {
                return -2;
            }
        }
    }
    ''',
    libraries=["webrtc-audio-processing-1"],
    include_dirs=["/usr/include/webrtc-audio-processing-1"],
    source_extension='.cpp',
    extra_compile_args=['-g', '-O0'],
    extra_link_args=['-g']
)

# Compile the C wrapper code. This will produce a Python extension module named "_webrtc_apm".
ffi.compile(verbose=False)

# Import the compiled extension module
import _webrtc_apm
print("Successfully imported _webrtc_apm")

# Initialize the WebRTC AudioProcessing module
print("Creating APM instance...")
apm = _webrtc_apm.lib.apm_create()
print(f"APM instance created: {apm != ffi.NULL}")
if apm == ffi.NULL:
    raise RuntimeError("Failed to create AudioProcessing instance")

print("Configuring APM...")
_webrtc_apm.lib.apm_configure(apm)
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
            
            # Debug CFFI buffer creation
            print("Creating CFFI buffers...")
            far_buf = ffi.from_buffer("float[]", far_mono)
            near_buf = ffi.from_buffer("float[]", near_mono)
            out_cbuf = ffi.from_buffer("float[]", out_buf)
            
            # Debug processing
            print("Calling apm_process...")
            err = _webrtc_apm.lib.apm_process(apm, far_buf, near_buf, out_cbuf,
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
    if apm != ffi.NULL:
        _webrtc_apm.lib.apm_destroy(apm)
        print("APM destroyed")
    print("Finished echo cancellation processing.")