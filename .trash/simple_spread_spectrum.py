import numpy as np
import wave
import struct
import hashlib
import os
import sys
from typing import Tuple, Optional, Union

class SimpleSpreadSpectrumSteganography:
    """
    Simple Spread Spectrum Audio Steganography System
    
    Uses only basic libraries (numpy, wave) for maximum compatibility.
    Hides binary data in WAV audio files using spread spectrum techniques.
    """
    
    def __init__(self, 
                 chip_rate: int = 512,  # Number of chips per bit
                 power_ratio: float = 0.001,  # Power ratio of hidden signal to host
                 sample_rate: int = 44100):
        
        self.chip_rate = chip_rate
        self.power_ratio = power_ratio
        self.sample_rate = sample_rate
        
        # Generate pseudo-random spreading sequence
        np.random.seed(42)  # Fixed seed for reproducibility
        self.spreading_sequence = np.random.choice([-1, 1], size=chip_rate).astype(np.float32)
    
    def load_wav(self, filename: str) -> Tuple[np.ndarray, int]:
        """Load WAV file and return audio data and sample rate"""
        with wave.open(filename, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            
            # Convert to numpy array
            if sampwidth == 1:
                dtype = np.uint8
                audio_data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                audio_data = (audio_data - 128) / 128.0  # Convert to [-1, 1]
            elif sampwidth == 2:
                dtype = np.int16
                audio_data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                audio_data = audio_data / 32768.0  # Convert to [-1, 1]
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")
            
            # Convert to mono if stereo
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2)
                audio_data = np.mean(audio_data, axis=1)
            
            return audio_data, sample_rate
    
    def save_wav(self, filename: str, audio_data: np.ndarray, sample_rate: int):
        """Save audio data to WAV file"""
        # Convert to 16-bit integers
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    def _prepare_data(self, data: bytes) -> Tuple[np.ndarray, int]:
        """Convert binary data to bits and add header information"""
        # Create header with data length and checksum
        data_length = len(data)
        checksum = hashlib.md5(data).digest()[:4]  # Use first 4 bytes of MD5
        
        # Pack header: length (4 bytes) + checksum (4 bytes)
        header = struct.pack('<I', data_length) + checksum
        full_data = header + data
        
        # Convert to bits
        bits = []
        for byte in full_data:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)
        
        return np.array(bits, dtype=np.float32), data_length
    
    def _bits_to_data(self, bits: np.ndarray) -> bytes:
        """Convert bits back to binary data"""
        # Ensure we have complete bytes
        if len(bits) % 8 != 0:
            padding = 8 - (len(bits) % 8)
            bits = np.concatenate([bits, np.zeros(padding)])
        
        bytes_data = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            byte_val = 0
            for j, bit in enumerate(byte_bits):
                if bit > 0.5:  # Threshold for bit detection
                    byte_val |= (1 << (7-j))
            bytes_data.append(byte_val)
        
        return bytes(bytes_data)
    
    def _spread_bits(self, bits: np.ndarray) -> np.ndarray:
        """Spread bits using pseudo-random sequence"""
        spread_signal = []
        for bit in bits:
            # Map bit: 0 -> -1, 1 -> +1
            bit_val = 2 * bit - 1
            # Spread the bit
            chip_sequence = bit_val * self.spreading_sequence
            spread_signal.extend(chip_sequence)
        
        return np.array(spread_signal, dtype=np.float32)
    
    def _despread_signal(self, signal_chips: np.ndarray) -> np.ndarray:
        """Despread signal to recover bits"""
        num_bits = len(signal_chips) // self.chip_rate
        bits = []
        
        for i in range(num_bits):
            start_idx = i * self.chip_rate
            end_idx = start_idx + self.chip_rate
            
            if end_idx <= len(signal_chips):
                chip_segment = signal_chips[start_idx:end_idx]
                # Correlate with spreading sequence
                correlation = np.dot(chip_segment, self.spreading_sequence)
                # Decide bit based on correlation sign
                bit = 1 if correlation > 0 else 0
                bits.append(bit)
        
        return np.array(bits, dtype=np.float32)
    
    def hide_data(self, host_audio: np.ndarray, data: bytes) -> np.ndarray:
        """
        Hide binary data in host audio using spread spectrum
        
        Args:
            host_audio: Host audio signal (normalized to [-1, 1])
            data: Binary data to hide
            
        Returns:
            Stego audio with hidden data
        """
        # Prepare data for hiding
        bits, original_length = self._prepare_data(data)
        
        # Spread the bits
        spread_signal = self._spread_bits(bits)
        
        # Calculate required audio length
        required_samples = len(spread_signal)
        
        if len(host_audio) < required_samples:
            raise ValueError(f"Host audio too short. Need {required_samples} samples, "
                           f"got {len(host_audio)}")
        
        # Create copy of host audio
        stego_audio = host_audio.copy()
        
        # Calculate adaptive power based on local signal energy
        for i in range(len(spread_signal)):
            local_power = host_audio[i] ** 2
            adaptive_power = self.power_ratio * (1 + local_power)
            stego_audio[i] += spread_signal[i] * adaptive_power
        
        return stego_audio
    
    def extract_data(self, stego_audio: np.ndarray, 
                    original_audio: Optional[np.ndarray] = None) -> Optional[bytes]:
        """
        Extract hidden data from stego audio
        
        Args:
            stego_audio: Audio with hidden data
            original_audio: Original host audio (optional, for better extraction)
            
        Returns:
            Extracted binary data or None if extraction failed
        """
        try:
            if original_audio is not None:
                # Subtract original to get hidden signal
                hidden_signal = stego_audio[:len(original_audio)] - original_audio
            else:
                # Use the stego audio directly (less reliable)
                hidden_signal = stego_audio.copy()
            
            # First, extract header to get data length
            header_bits = 64  # 8 bytes * 8 bits = 64 bits for header
            header_chips = header_bits * self.chip_rate
            
            if len(hidden_signal) < header_chips:
                return None
            
            # Extract header
            header_signal = hidden_signal[:header_chips]
            header_bits_recovered = self._despread_signal(header_signal)
            header_bytes = self._bits_to_data(header_bits_recovered)
            
            if len(header_bytes) < 8:
                return None
            
            # Parse header
            data_length = struct.unpack('<I', header_bytes[:4])[0]
            expected_checksum = header_bytes[4:8]
            
            # Validate data length
            if data_length > 1000000:  # Sanity check: max 1MB
                return None
            
            # Calculate total bits needed
            total_bits = header_bits + (data_length * 8)
            total_chips = total_bits * self.chip_rate
            
            if len(hidden_signal) < total_chips:
                return None
            
            # Extract all data
            all_signal = hidden_signal[:total_chips]
            all_bits_recovered = self._despread_signal(all_signal)
            all_bytes = self._bits_to_data(all_bits_recovered)
            
            # Extract actual data (skip header)
            if len(all_bytes) < 8 + data_length:
                return None
            
            extracted_data = all_bytes[8:8+data_length]
            
            # Verify checksum
            calculated_checksum = hashlib.md5(extracted_data).digest()[:4]
            
            if calculated_checksum == expected_checksum:
                return extracted_data
            else:
                print("Warning: Checksum mismatch - data may be corrupted")
                return extracted_data  # Return anyway, might be partially correct
                
        except Exception as e:
            print(f"Extraction error: {e}")
            return None
    
    def hide_file(self, host_audio_path: str, 
                  data_file_path: str, 
                  output_path: str) -> bool:
        """
        Hide a file in audio and save the result
        
        Args:
            host_audio_path: Path to host WAV file
            data_file_path: Path to file to hide
            output_path: Path for output stego WAV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load host audio
            host_audio, sr = self.load_wav(host_audio_path)
            
            # Read data file
            with open(data_file_path, 'rb') as f:
                data = f.read()
            
            print(f"Hiding {len(data)} bytes in audio...")
            
            # Hide data
            stego_audio = self.hide_data(host_audio, data)
            
            # Save stego audio
            self.save_wav(output_path, stego_audio, sr)
            
            print(f"Stego audio saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error hiding file: {e}")
            return False
    
    def extract_file(self, stego_audio_path: str, 
                    output_file_path: str,
                    original_audio_path: Optional[str] = None) -> bool:
        """
        Extract hidden file from stego audio
        
        Args:
            stego_audio_path: Path to stego WAV file
            output_file_path: Path for extracted file
            original_audio_path: Path to original WAV file (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load stego audio
            stego_audio, sr = self.load_wav(stego_audio_path)
            
            # Load original audio if provided
            original_audio = None
            if original_audio_path:
                original_audio, _ = self.load_wav(original_audio_path)
            
            print("Extracting hidden data...")
            
            # Extract data
            extracted_data = self.extract_data(stego_audio, original_audio)
            
            if extracted_data is not None:
                # Save extracted data
                with open(output_file_path, 'wb') as f:
                    f.write(extracted_data)
                
                print(f"Extracted {len(extracted_data)} bytes to: {output_file_path}")
                return True
            else:
                print("Failed to extract data")
                return False
                
        except Exception as e:
            print(f"Error extracting file: {e}")
            return False


def create_test_audio(filename: str, duration: float = 10.0, sample_rate: int = 44100):
    """Create a test audio file"""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Create a simple melody with some complexity
    frequencies = [220, 247, 262, 294, 330, 349, 392]  # Musical notes
    audio = np.zeros(samples)
    
    # Add multiple frequency components
    for i, freq in enumerate(frequencies):
        phase = i * np.pi / 4  # Different phase for each component
        amplitude = 0.1 * (1 + i) / len(frequencies)  # Varying amplitudes
        audio += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Add some gentle noise for realism
    audio += 0.02 * np.random.randn(samples)
    
    # Apply envelope
    envelope = np.exp(-t * 0.05)  # Slow decay
    audio *= envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.7
    
    # Save as WAV
    steg = SimpleSpreadSpectrumSteganography()
    steg.save_wav(filename, audio, sample_rate)
    print(f"Test audio created: {filename}")


def demo_simple_steganography():
    """Demonstration of the simple steganography system"""
    print("SIMPLE SPREAD SPECTRUM STEGANOGRAPHY DEMO")
    print("="*50)
    
    # Create steganography system with better capacity
    steg = SimpleSpreadSpectrumSteganography(
        chip_rate=128,      # Smaller chip rate for more capacity
        power_ratio=0.01    # Higher for robustness
    )
    
    # Create test audio - much longer for more capacity
    host_audio_path = "/tmp/test_host.wav"
    create_test_audio(host_audio_path, duration=60.0)  # 60 seconds
    
    # Create test data files
    test_files = {}
    
    # Text file
    text_content = "This is a secret message hidden in audio using spread spectrum steganography! " * 5
    text_path = "/tmp/secret_message.txt"
    with open(text_path, 'w') as f:
        f.write(text_content)
    test_files['text'] = text_path
    
    # Binary file (fake image data)
    binary_data = b'FAKE_IMAGE_HEADER' + b'\x00\x01\x02\x03' * 100
    binary_path = "/tmp/secret_image.bin"
    with open(binary_path, 'wb') as f:
        f.write(binary_data)
    test_files['binary'] = binary_path
    
    # Test each file type
    for file_type, file_path in test_files.items():
        print(f"\n--- Testing {file_type.upper()} file ---")
        
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")
        
        # Hide file
        stego_path = f"/tmp/stego_{file_type}.wav"
        success = steg.hide_file(host_audio_path, file_path, stego_path)
        
        if success:
            print(f"✅ Successfully hidden {file_type} file")
            
            # Extract file (with original for better results)
            extracted_path = f"/tmp/extracted_{file_type}.{'txt' if file_type == 'text' else 'bin'}"
            extract_success = steg.extract_file(stego_path, extracted_path, host_audio_path)
            
            if extract_success:
                extracted_size = os.path.getsize(extracted_path)
                print(f"✅ Successfully extracted: {extracted_size} bytes")
                
                # Verify extraction
                with open(file_path, 'rb') as f:
                    original_data = f.read()
                with open(extracted_path, 'rb') as f:
                    extracted_data = f.read()
                
                if original_data == extracted_data:
                    print("🎉 Perfect extraction!")
                else:
                    print("⚠️ Some data corruption detected")
                
                # For text files, show content
                if file_type == 'text':
                    try:
                        with open(extracted_path, 'r') as f:
                            extracted_text = f.read()
                        print(f"Extracted text preview: {extracted_text[:100]}...")
                    except:
                        pass
            else:
                print("❌ Extraction failed")
        else:
            print(f"❌ Failed to hide {file_type} file")
    
    print(f"\n--- Generated files in /tmp/ ---")
    for file in sorted(os.listdir('/tmp')):
        if any(file.startswith(prefix) for prefix in ['test_', 'secret_', 'stego_', 'extracted_']):
            print(f"  {file}")
    
    print("\n🎯 DEMO COMPLETE!")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Command line usage: python script.py host_audio.wav data_file.txt
        host_audio_path = sys.argv[1]
        data_file_path = sys.argv[2]
        
        print(f"HOST AUDIO: {host_audio_path}")
        print(f"DATA FILE: {data_file_path}")
        print("="*50)
        
        # Check if files exist
        if not os.path.exists(host_audio_path):
            print(f"❌ Host audio file not found: {host_audio_path}")
            sys.exit(1)
        
        if not os.path.exists(data_file_path):
            print(f"❌ Data file not found: {data_file_path}")
            sys.exit(1)
        
        # Get file sizes and estimate capacity
        try:
            host_audio, sr = SimpleSpreadSpectrumSteganography().load_wav(host_audio_path)
            data_size = os.path.getsize(data_file_path)
            
            print(f"Host audio: {len(host_audio)} samples, {len(host_audio)/sr:.1f} seconds")
            print(f"Data file: {data_size} bytes")
            
            # Calculate required capacity
            # Each bit needs chip_rate samples, plus header (8 bytes = 64 bits)
            chip_rate = 64  # Use smaller chip rate for more capacity
            total_bits = (data_size + 8) * 8  # +8 for header
            required_samples = total_bits * chip_rate
            
            print(f"Required samples: {required_samples}")
            print(f"Available samples: {len(host_audio)}")
            
            if required_samples > len(host_audio):
                print("⚠️ Not enough capacity with default settings. Trying optimized parameters...")
                
                # Try with even smaller chip rate
                chip_rate = 32
                required_samples = total_bits * chip_rate
                print(f"With chip_rate={chip_rate}: {required_samples} samples needed")
                
                if required_samples > len(host_audio):
                    chip_rate = 16
                    required_samples = total_bits * chip_rate
                    print(f"With chip_rate={chip_rate}: {required_samples} samples needed")
            
            if required_samples <= len(host_audio):
                print(f"✅ Using chip_rate={chip_rate}")
                
                # Create steganography system with optimized parameters
                steg = SimpleSpreadSpectrumSteganography(
                    chip_rate=chip_rate,
                    power_ratio=0.01  # Higher for robustness
                )
                
                # Hide data
                base_name = os.path.splitext(os.path.basename(host_audio_path))[0]
                stego_path = f"{base_name}_stego.wav"
                
                print(f"\nHiding data...")
                success = steg.hide_file(host_audio_path, data_file_path, stego_path)
                
                if success:
                    print(f"✅ Stego audio saved: {stego_path}")
                    
                    # Test extraction
                    print("\nTesting extraction...")
                    extracted_path = f"extracted_{os.path.basename(data_file_path)}"
                    extract_success = steg.extract_file(stego_path, extracted_path, host_audio_path)
                    
                    if extract_success:
                        print(f"✅ Data extracted to: {extracted_path}")
                        
                        # Verify
                        with open(data_file_path, 'rb') as f:
                            original = f.read()
                        with open(extracted_path, 'rb') as f:
                            extracted = f.read()
                        
                        if original == extracted:
                            print("🎉 Perfect extraction verified!")
                        else:
                            print("⚠️ Some differences detected")
                    else:
                        print("❌ Extraction failed")
                else:
                    print("❌ Failed to hide data")
            else:
                print("❌ Host audio too short even with minimum chip rate")
                print(f"Need at least {required_samples/sr:.1f} seconds of audio")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif len(sys.argv) >= 4 and sys.argv[1] == "extract":
        # Extract mode: python script.py extract stego.wav output_file [original.wav]
        stego_path = sys.argv[2] 
        output_path = sys.argv[3]
        original_path = None
        
        if len(sys.argv) >= 5:
            original_path = sys.argv[4]
        
        print(f"EXTRACTION MODE")
        print(f"Stego audio: {stego_path}")
        print(f"Output file: {output_path}")
        if original_path:
            print(f"Original audio: {original_path}")
        print("="*50)
        
        # Try different chip rates to find the right one
        for chip_rate in [16, 32, 64, 128, 256, 512]:
            print(f"Trying chip_rate={chip_rate}...")
            
            steg = SimpleSpreadSpectrumSteganography(
                chip_rate=chip_rate,
                power_ratio=0.01
            )
            
            success = steg.extract_file(stego_path, output_path, original_path)
            if success:
                print(f"✅ Extraction successful with chip_rate={chip_rate}")
                break
        else:
            print("❌ Extraction failed with all chip rates")
    
    else:
        demo_simple_steganography()
    
        print("\n" + "="*60)
        print("COMMAND LINE USAGE:")
        print("="*60)
        print("Hide data:")
        print("  python script.py host_audio.wav data_file.txt")
        print()
        print("Extract data:")
        print("  python script.py extract stego_audio.wav output_file.txt [original_audio.wav]")
        print()
        print("USAGE EXAMPLES:")
        print("="*60)
        
        print("""
# Basic usage:
steg = SimpleSpreadSpectrumSteganography()

# Hide a file in audio
steg.hide_file('host.wav', 'secret.txt', 'stego.wav')

# Extract the hidden file
steg.extract_file('stego.wav', 'extracted.txt', 'host.wav')

# Direct data hiding:
host_audio, sr = steg.load_wav('host.wav')
with open('data.bin', 'rb') as f:
    data = f.read()

stego_audio = steg.hide_data(host_audio, data)
steg.save_wav('stego.wav', stego_audio, sr)

# Direct data extraction:
stego_audio, sr = steg.load_wav('stego.wav')
host_audio, _ = steg.load_wav('host.wav')
extracted_data = steg.extract_data(stego_audio, host_audio)
        """)
