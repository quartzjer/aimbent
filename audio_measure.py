import soundcard as sc
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from scipy import signal
import os
from collections import deque
import datetime

def generate_ultrasonic_signal(sample_rate, duration=0.005, frequency=20500):
    """Generate a short ultrasonic tone (inaudible to humans)"""
    # Generate the ultrasonic tone (above human hearing range)
    t = np.linspace(0, duration, int(sample_rate * duration))
    tone = np.sin(2 * np.pi * frequency * t)
    
    # Apply a window to avoid clicks
    window = np.hamming(len(tone))
    tone = tone * window
    
    # Normalize amplitude
    tone = tone / np.max(np.abs(tone)) * 0.9
    
    # Add silent padding at the beginning
    padding = np.zeros(int(sample_rate * 0.001))  # 1ms of silence
    test_signal = np.concatenate((padding, tone))
    
    return test_signal

class TestSignalPlayer:
    def __init__(self, speaker, sample_rate, frequency=20500):
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.test_signal = generate_ultrasonic_signal(sample_rate, frequency=frequency)
        self.running = False
        self.play_thread = None
        
    def start(self, interval=0.1):
        """Start playing the test signal repeatedly"""
        if self.running:
            return
            
        self.running = True
        self.play_thread = threading.Thread(target=self._play_loop, args=(interval,))
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def stop(self):
        """Stop playing the test signal"""
        self.running = False
        if self.play_thread:
            self.play_thread.join()
            self.play_thread = None
    
    def _play_loop(self, interval):
        """Continuously play the test signal at specified intervals"""
        while self.running:
            play_start = time.perf_counter()
            #self.speaker.play(self.test_signal, self.sample_rate)
            elapsed = time.perf_counter() - play_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

def single_latency_test(signal_player, mic, loopback, sample_rate, recording_duration=0.5):
    """Run a single latency test using an ultrasonic tone"""
    # Create recording threads for both devices
    mic_recording = None
    loopback_recording = None
    
    def record_mic():
        nonlocal mic_recording
        mic_recording = mic.record(samplerate=sample_rate, numframes=int(recording_duration * sample_rate))
    
    def record_loopback():
        nonlocal loopback_recording
        loopback_recording = loopback.record(samplerate=sample_rate, numframes=int(recording_duration * sample_rate))
    
    # Start recording threads
    mic_thread = threading.Thread(target=record_mic)
    loopback_thread = threading.Thread(target=record_loopback)
    
    mic_thread.start()
    loopback_thread.start()
    
    # Wait for recordings to complete
    mic_thread.join()
    loopback_thread.join()
    
    # Calculate latency
    latency, correlation_strength, proc_time = calculate_latency_from_recordings(mic_recording, loopback_recording, sample_rate)
    
    return latency, correlation_strength, proc_time

def calculate_latency_from_recordings(mic_recording, loopback_recording, sample_rate):
    """Calculate the latency between loopback and mic recordings"""
    calc_start_time = time.perf_counter()
    
    # Convert stereo to mono if needed
    if len(loopback_recording.shape) > 1 and loopback_recording.shape[1] > 1:
        loopback_mono = np.mean(loopback_recording, axis=1)
    else:
        loopback_mono = loopback_recording.flatten()
        
    if len(mic_recording.shape) > 1 and mic_recording.shape[1] > 1:
        mic_mono = np.mean(mic_recording, axis=1)
    else:
        mic_mono = mic_recording.flatten()
    
    # Compute cross-correlation to find delay
    correlation = signal.correlate(mic_mono, loopback_mono, mode='full')
    lags = signal.correlation_lags(len(mic_mono), len(loopback_mono), mode='full')
    max_idx = np.argmax(correlation)
    lag = lags[max_idx]
    
    # To verify the quality of the correlation, check if it's significantly strong
    correlation_strength = correlation[max_idx] / np.mean(correlation)
    
    # Convert lag to seconds
    latency = lag / sample_rate
    
    # Calculate processing time
    processing_time = (time.perf_counter() - calc_start_time) * 1000  # Convert to milliseconds
    
    return latency, correlation_strength, processing_time

def continuous_latency_monitoring(signal_player, mic, loopback, sample_rate, interval=1.0, duration=60, plot_update_interval=5):
    """Monitor latency continuously over time"""
    print(f"\nStarting continuous latency monitoring")
    print(f"Using ultrasonic tone (not audible to humans)")
    print(f"Tests will run every {interval} seconds for {duration} seconds")
    print(f"Press Ctrl+C to stop monitoring early")
    
    # Start the signal player if it's not already running
    signal_player.start()
    
    # Create deques to store monitoring data
    timestamps = deque(maxlen=int(duration/interval) + 10)
    latencies = deque(maxlen=int(duration/interval) + 10)
    correlation_strengths = deque(maxlen=int(duration/interval) + 10)
    proc_times = deque(maxlen=int(duration/interval) + 10)
    
    # Create a figure for real-time plotting
    plt.figure(figsize=(12, 6))
    # Remove plt.ion() - we don't want interactive mode
    
    start_time = time.time()
    last_plot_update = start_time
    
    try:
        # Run tests at specified intervals
        while time.time() - start_time < duration:
            test_start = time.time()
            
            # Run a single latency test
            latency, corr_strength, proc_time = single_latency_test(signal_player, mic, loopback, sample_rate)
            
            # Record the results
            timestamps.append(time.time() - start_time)
            latencies.append(latency * 1000)  # Convert to ms
            correlation_strengths.append(corr_strength)
            proc_times.append(proc_time)
            
            # Print current latency
            print(f"Time: {timestamps[-1]:.1f}s - Latency: {latencies[-1]:.2f}ms (Correlation strength: {correlation_strengths[-1]:.2f}, Processing time: {proc_time:.2f}ms)")
            
            # Wait until next interval
            elapsed = time.time() - test_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    except KeyboardInterrupt:
        print("\nLatency monitoring stopped by user")
    finally:
        # Stop the signal player
        signal_player.stop()
    
    # Final plot update with final=True
    update_latency_plot(timestamps, latencies, correlation_strengths, final=True)
    
    # Save results to CSV
    save_results_to_csv(timestamps, latencies, correlation_strengths, proc_times)
    
    # Print summary statistics
    print_latency_statistics(latencies)
    
    return list(timestamps), list(latencies), list(correlation_strengths)

def update_latency_plot(timestamps, latencies, correlation_strengths, final=False):
    """Update the real-time latency plot"""
    plt.clf()
    
    # Create a color gradient based on correlation strength
    colors = []
    for strength in correlation_strengths:
        if strength > 5:  # Good correlation
            colors.append('green')
        elif strength > 3:  # Moderate correlation
            colors.append('orange')
        else:  # Poor correlation
            colors.append('red')
    
    # Plot latency over time
    plt.scatter(list(timestamps), list(latencies), c=colors, alpha=0.7)
    plt.plot(list(timestamps), list(latencies), 'b-', alpha=0.3)
    
    # Add a trend line
    if len(timestamps) > 2:
        z = np.polyfit(list(timestamps), list(latencies), 1)
        p = np.poly1d(z)
        plt.plot(list(timestamps), p(list(timestamps)), "r--", alpha=0.8)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Latency (ms)')
    plt.title('Audio System Latency Over Time')
    plt.grid(True)
    
    # Add a horizontal line for the average latency
    if latencies:
        avg_latency = np.mean(latencies)
        plt.axhline(y=avg_latency, color='g', linestyle='--', alpha=0.8)
        plt.text(min(timestamps), avg_latency, f' Avg: {avg_latency:.2f}ms', verticalalignment='bottom')
    
    plt.tight_layout()
    
    if final:
        # Save the final plot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'latency_monitoring_{timestamp}.png'
        plt.savefig(filename)
        print(f"Latency graph saved to '{filename}'")
        plt.show()  # Only show the plot at the end

def save_results_to_csv(timestamps, latencies, correlation_strengths, proc_times):
    """Save the monitoring results to a CSV file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'latency_data_{timestamp}.csv'
    
    with open(filename, 'w') as f:
        f.write("Time (s),Latency (ms),Correlation_Strength,Processing_Time (ms)\n")
        for t, l, c, p in zip(timestamps, latencies, correlation_strengths, proc_times):
            f.write(f"{t:.2f},{l:.2f},{c:.2f},{p:.2f}\n")
    
    print(f"Latency data saved to '{filename}'")

def print_latency_statistics(latencies):
    """Print summary statistics for the latency measurements"""
    if not latencies:
        return
    
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)
    
    print("\nLatency Statistics:")
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"Minimum latency: {min_latency:.2f}ms")
    print(f"Maximum latency: {max_latency:.2f}ms")
    print(f"Standard deviation: {std_latency:.2f}ms")
    print(f"Jitter (max-min): {max_latency - min_latency:.2f}ms")

def main():
    # List all microphones including loopback (monitor) devices
    mics = sc.all_microphones(include_loopback=True)
    if len(mics) < 2:
        print("Error: Need at least two audio devices (mic and loopback)!")
        return

    print("Available audio devices:")
    for i, mic in enumerate(mics):
        print(f"{i}: {mic.name}")
    
    try:
        # Use the first two devices as specified in the requirements
        mic_id = 0  # First device (microphone input)
        loopback_id = 1  # Second device (system loopback)
        
        print(f"\nUsing device {mic_id} as microphone: {mics[mic_id].name}")
        print(f"Using device {loopback_id} as loopback: {mics[loopback_id].name}")
        
        # Get default speaker
        speaker = sc.default_speaker()
        print(f"Using default speaker: {speaker.name}")
        
        # Set sample rate (using 48kHz for better ultrasonic reproduction)
        sample_rate = 16000
        
        # Create the test signal player
        signal_player = TestSignalPlayer(speaker, sample_rate)
        
        while True:
            print("\nAudio Latency Measurement Options:")
            print("1. Single test (one-time measurement)")
            print("2. Continuous monitoring (measure latency over time)")
            print("3. Exit")
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1':
                # Start the signal player
                signal_player.start()
                # Run a single test
                latency, corr_strength, proc_time = single_latency_test(signal_player, mics[mic_id], mics[loopback_id], sample_rate)
                # Stop the signal player
                signal_player.stop()
                print(f"\nDetected latency: {latency*1000:.2f} milliseconds")
                print(f"Correlation strength: {corr_strength:.2f}")
                print(f"Processing time: {proc_time:.2f} milliseconds")
                
            elif choice == '2':
                # Start continuous monitoring
                duration = 60  # Default monitoring duration in seconds
                try:
                    user_duration = input("Enter monitoring duration in seconds (default: 60): ")
                    if user_duration.strip():
                        duration = float(user_duration)
                except ValueError:
                    print("Invalid input, using default 60 seconds")
                
                interval = 1.0  # Default interval between tests
                try:
                    user_interval = input("Enter interval between tests in seconds (default: 1.0): ")
                    if user_interval.strip():
                        interval = float(user_interval)
                except ValueError:
                    print("Invalid input, using default 1.0 second interval")
                
                # Run continuous monitoring (signal player is managed inside the function)
                continuous_latency_monitoring(
                    signal_player, mics[mic_id], mics[loopback_id], sample_rate,
                    interval=interval, duration=duration
                )
                
            elif choice == '3':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice, please try again.")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()