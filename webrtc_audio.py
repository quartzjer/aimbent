from cffi import FFI
import numpy as np

class WebRTCAudioError(Exception):
    """Base exception for WebRTC audio processing errors"""
    pass

class WebRTCAudio:
    def __init__(self):
        self.ffi = FFI()
        self.ffi.cdef("""
            typedef struct AudioProcessing AudioProcessing;
            AudioProcessing* apm_create();
            void apm_destroy(AudioProcessing* apm);
            void apm_configure(AudioProcessing* apm);
            int apm_process(AudioProcessing* apm,
                            const float* far_frame,
                            const float* near_frame,
                            float* out_frame,
                            int rate, int channels, int frame_size);
        """)
        self.ffi.set_source(
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
                        } catch (...) {}
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
                        
                        if (far_frame) {
                            const float* const far_channels[1] = { far_frame };
                            float far_out[480];
                            float* far_out_channels[1] = { far_out };
                            
                            int err = apm->ProcessReverseStream(far_channels, stream_config, stream_config, far_out_channels);
                            if (err != 0) {
                                return -4;
                            }
                        }
                        
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
        self.ffi.compile(verbose=False)
        import _webrtc_apm
        self.lib = _webrtc_apm.lib

    def create_apm(self):
        apm = self.lib.apm_create()
        if apm == self.ffi.NULL:
            raise RuntimeError("Failed to create AudioProcessing instance")
        return apm

    def configure_apm(self, apm):
        self.lib.apm_configure(apm)

    def process_frame(self, apm, far_frame, near_frame, sample_rate=48000, channels=1, frame_size=480):
        """
        Process audio frames through WebRTC APM.
        
        Args:
            apm: APM instance
            far_frame: Far-end (reference) audio as numpy array
            near_frame: Near-end audio as numpy array
            sample_rate: Sample rate (default: 48000)
            channels: Number of channels (default: 1)
            frame_size: Frame size (default: 480)
            
        Returns:
            numpy.ndarray: Processed audio frame
            
        Raises:
            WebRTCAudioError: If processing fails
        """
        if not isinstance(far_frame, np.ndarray) or not isinstance(near_frame, np.ndarray):
            raise TypeError("Audio frames must be numpy arrays")
            
        out_frame = np.zeros(frame_size, dtype=np.float32)
        
        far_buf = self.ffi.from_buffer("float[]", far_frame)
        near_buf = self.ffi.from_buffer("float[]", near_frame)
        out_buf = self.ffi.from_buffer("float[]", out_frame)
        
        result = self.lib.apm_process(apm, far_buf, near_buf, out_buf, 
                                    sample_rate, channels, frame_size)
        
        if result != 0:
            error_messages = {
                -1: "Invalid parameters",
                -2: "Processing exception",
                -3: "Invalid sample rate/channels/frame size configuration",
                -4: "Error processing far-end stream"
            }
            raise WebRTCAudioError(error_messages.get(result, f"Unknown error: {result}"))
            
        return out_frame

    def destroy_apm(self, apm):
        self.lib.apm_destroy(apm)
