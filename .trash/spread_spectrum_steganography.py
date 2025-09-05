import numpy as np
import hashlib
import struct
from typing import Tuple, Optional

class SpreadSpectrumCore:
    """
    Spread Spectrum Audio Steganography System
    
    This system hides binary data in audio files using spread spectrum techniques,
    making the hidden data robust to moderate compression.
    """
    
    def __init__(self, 
                 chip_rate: int = 1024,  # Number of chips per bit
                 power_ratio: float = 0.001,  # Power ratio of hidden signal to host
                 frame_size: int = 2048,  # Frame size for processing
                 overlap: float = 0.5):  # Frame overlap ratio
        
        self.chip_rate = chip_rate
        self.power_ratio = power_ratio
        self.frame_size = frame_size
        self.overlap = overlap
        self.hop_length = int(frame_size * (1 - overlap))
        
        # Generate pseudo-random spreading sequence
        np.random.seed(42)  # Fixed seed for reproducibility
        self.spreading_sequence = np.random.choice([-1, 1], size=chip_rate)
        
    def _generate_pn_sequence(self, length: int, seed: int = 42) -> np.ndarray:
        """Generate pseudo-random noise sequence for spreading"""
        np.random.seed(seed)
        return np.random.choice([-1, 1], size=length)
    
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
    
    def _bits_to_data(self, bits: np.ndarray, expected_length: int) -> bytes:
        """Convert bits back to binary data with validation"""
        # Convert bits to bytes
        if len(bits) % 8 != 0:
            # Pad with zeros if necessary
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
    
    def _adaptive_power_adjustment(self, host_frame: np.ndarray, 
                                 hidden_frame: np.ndarray) -> np.ndarray:
        """Adaptively adjust hidden signal power based on host signal characteristics"""
        # Calculate host signal energy
        host_energy = np.mean(host_frame ** 2)
        
        if host_energy > 0:
            # Adjust power based on local signal characteristics
            # Higher energy regions can hide more data
            local_power_ratio = self.power_ratio * (1 + np.sqrt(host_energy))
            hidden_frame = hidden_frame * np.sqrt(local_power_ratio * host_energy)
        
        return hidden_frame
    
    def hide_data(self, host_audio: np.ndarray, 
                  data: bytes, 
                  sample_rate: int = 44100) -> np.ndarray:
        """
        Hide binary data in host audio using spread spectrum
        
        Args:
            host_audio: Host audio signal
            data: Binary data to hide
            sample_rate: Audio sample rate
            
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
        
        # Process in frames for better imperceptibility
        for i in range(0, len(spread_signal), self.frame_size):
            frame_end = min(i + self.frame_size, len(spread_signal))
            host_frame = host_audio[i:frame_end]
            hidden_frame = spread_signal[i:frame_end]
            
            # Adaptive power adjustment
            hidden_frame = self._adaptive_power_adjustment(host_frame, hidden_frame)
            
            # Add hidden signal to host
            stego_audio[i:frame_end] += hidden_frame
        
        return stego_audio
    
    def extract_data(self, stego_audio: np.ndarray, 
                    original_audio: Optional[np.ndarray] = None,
                    sample_rate: int = 44100) -> Optional[bytes]:
        """
        Extract hidden data from stego audio
        
        Args:
            stego_audio: Audio with hidden data
            original_audio: Original host audio (optional, for better extraction)
            sample_rate: Audio sample rate
            
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
            header_bytes = self._bits_to_data(header_bits_recovered, 8)
            
            if len(header_bytes) < 8:
                return None
            
            # Parse header
            data_length = struct.unpack('<I', header_bytes[:4])[0]
            expected_checksum = header_bytes[4:8]
            
            # Calculate total bits needed
            total_bits = header_bits + (data_length * 8)
            total_chips = total_bits * self.chip_rate
            
            if len(hidden_signal) < total_chips:
                return None
            
            # Extract all data
            all_signal = hidden_signal[:total_chips]
            all_bits_recovered = self._despread_signal(all_signal)
            all_bytes = self._bits_to_data(all_bits_recovered, len(all_bits_recovered) // 8)
            
            # Extract actual data (skip header)
            if len(all_bytes) < 8 + data_length:
                return None
            
            extracted_data = all_bytes[8:8+data_length]
            
            # Verify checksum
            calculated_checksum = hashlib.md5(extracted_data).digest()[:4]
            
            if calculated_checksum == expected_checksum:
                return extracted_data
            else:
                print("Checksum mismatch - data may be corrupted")
                return extracted_data  # Return anyway, might be partially correct
                
        except Exception as e:
            print(f"Extraction error: {e}")
            return None
    
    def hide_file(self, host_audio_path: str, 
                  data_file_path: str, 
                  output_path: str,
                  audio_format: str = 'wav') -> bool:
        """
        Hide a file in audio and save the result
        
        Args:
            host_audio_path: Path to host audio file
            data_file_path: Path to file to hide
            output_path: Path for output stego audio
            audio_format: Output audio format
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load host audio
            host_audio, sr = librosa.load(host_audio_path, sr=None)
            
            # Read data file
            with open(data_file_path, 'rb') as f:
                data = f.read()
            
            print(f"Hiding {len(data)} bytes in audio...")
            
            # Hide data
            stego_audio = self.hide_data(host_audio, data, sr)
            
            # Save stego audio
            sf.write(output_path, stego_audio, sr, format=audio_format.upper())
            
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
            stego_audio_path: Path to stego audio
            output_file_path: Path for extracted file
            original_audio_path: Path to original audio (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load stego audio
            stego_audio, sr = librosa.load(stego_audio_path, sr=None)
            
            # Load original audio if provided
            original_audio = None
            if original_audio_path:
                original_audio, _ = librosa.load(original_audio_path, sr=sr)
            
            print("Extracting hidden data...")
            
            # Extract data
            extracted_data = self.extract_data(stego_audio, original_audio, sr)
            
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


def demo_spread_spectrum_steganography():
    """Demonstration of the spread spectrum steganography system"""
    
    # Create steganography system
    steg_system = SpreadSpectrumSteganography(
        chip_rate=512,  # Smaller chip rate for demo
        power_ratio=0.002,  # Slightly higher power for robustness
        frame_size=1024
    )
    
    # Generate demo host audio (10 seconds of pink noise)
    duration = 10.0  # seconds
    sample_rate = 44100
    samples = int(duration * sample_rate)
    
    # Generate pink noise as host audio
    white_noise = np.random.randn(samples)
    # Simple pink noise filter
    b, a = signal.butter(1, 0.5, 'low')
    host_audio = signal.filtfilt(b, a, white_noise)
    host_audio = host_audio / np.max(np.abs(host_audio)) * 0.5  # Normalize
    
    # Create demo data to hide
    demo_text = "This is a secret message hidden using spread spectrum steganography! " * 10
    demo_data = demo_text.encode('utf-8')
    
    print(f"Demo: Hiding {len(demo_data)} bytes of text data")
    
    # Hide data
    try:
        stego_audio = steg_system.hide_data(host_audio, demo_data, sample_rate)
        print("Data hidden successfully!")
        
        # Extract data (with original audio for better results)
        extracted_data = steg_system.extract_data(stego_audio, host_audio, sample_rate)
        
        if extracted_data:
            extracted_text = extracted_data.decode('utf-8', errors='ignore')
            print(f"Extracted {len(extracted_data)} bytes")
            print(f"Original text length: {len(demo_text)}")
            print(f"Extracted text length: {len(extracted_text)}")
            
            # Check if extraction was successful
            if extracted_text.strip() == demo_text.strip():
                print("✅ Perfect extraction!")
            else:
                print("⚠️ Partial extraction - some data may be corrupted")
                print(f"First 100 chars of extracted: {extracted_text[:100]}...")
        else:
            print("❌ Extraction failed")
            
        # Save demo files
        sf.write('/tmp/demo_host.wav', host_audio, sample_rate)
        sf.write('/tmp/demo_stego.wav', stego_audio, sample_rate)
        print("Demo audio files saved to /tmp/")
        
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    # Run demonstration
    demo_spread_spectrum_steganography()
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES:")
    print("="*60)
    
    print("""
# Create steganography system
steg = SpreadSpectrumSteganography()

# Hide a text file in audio
steg.hide_file('host_audio.wav', 'secret.txt', 'stego_audio.wav')

# Extract the hidden file
steg.extract_file('stego_audio.wav', 'extracted_secret.txt', 'host_audio.wav')

# Hide binary data directly
with open('image.jpg', 'rb') as f:
    image_data = f.read()

host_audio, sr = librosa.load('host.wav', sr=None)
stego_audio = steg.hide_data(host_audio, image_data, sr)

# Extract binary data
extracted_data = steg.extract_data(stego_audio, host_audio, sr)
with open('extracted_image.jpg', 'wb') as f:
    f.write(extracted_data)
    """)
