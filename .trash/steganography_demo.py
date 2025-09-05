#!/usr/bin/env python3
"""
Comprehensive example of spread spectrum steganography
Demonstrates hiding text, images, and audio files in host audio
"""

import numpy as np
import librosa
import soundfile as sf
from spread_spectrum_steganography import SpreadSpectrumSteganography
import os
import tempfile

def create_test_files():
    """Create test files for demonstration"""
    test_files = {}
    
    # Create a test text file
    text_content = """
This is a comprehensive test of spread spectrum steganography.
The system can hide various types of binary data including:
- Text files (.txt)
- Images (.jpg, .png, .gif, etc.)
- Audio files (.wav, .mp3, etc.)
- Any binary file format

The spread spectrum technique spreads the hidden data across 
the frequency spectrum, making it robust to moderate compression
and difficult to detect without the proper extraction algorithm.

Key features:
1. Robust to moderate compression
2. Adaptive power adjustment
3. Error detection with checksums
4. Support for any binary file format
5. Configurable parameters for different use cases
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(text_content.strip())
        test_files['text'] = f.name
    
    # Create a small test image (as text representation)
    # In real use, you'd use actual image files
    fake_image_data = b'\x89PNG\r\n\x1a\n' + b'FAKE_IMAGE_DATA' * 100  # Simulate PNG header + data
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(fake_image_data)
        test_files['image'] = f.name
    
    # Create test audio data
    sample_rate = 22050
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a simple melody
    frequencies = [440, 523, 659, 784]  # A, C, E, G
    audio_data = np.zeros_like(t)
    for i, freq in enumerate(frequencies):
        start = i * len(t) // len(frequencies)
        end = (i + 1) * len(t) // len(frequencies)
        audio_data[start:end] = 0.3 * np.sin(2 * np.pi * freq * t[start:end])
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio_data, sample_rate)
        test_files['audio'] = f.name
    
    return test_files

def create_host_audio(duration=30.0, sample_rate=44100):
    """Create a host audio signal"""
    samples = int(duration * sample_rate)
    
    # Create a more realistic host audio (music-like)
    t = np.linspace(0, duration, samples)
    
    # Base melody
    melody_freq = 220  # A3
    melody = 0.3 * np.sin(2 * np.pi * melody_freq * t)
    
    # Add harmonics
    for harmonic in [2, 3, 4]:
        melody += 0.1 * np.sin(2 * np.pi * melody_freq * harmonic * t) / harmonic
    
    # Add some bass
    bass_freq = 110  # A2
    bass = 0.2 * np.sin(2 * np.pi * bass_freq * t)
    
    # Add noise for realism
    noise = 0.05 * np.random.randn(samples)
    
    # Combine all elements
    host_audio = melody + bass + noise
    
    # Apply envelope to make it more natural
    envelope = np.exp(-t * 0.1)  # Decay envelope
    host_audio = host_audio * envelope
    
    # Normalize
    host_audio = host_audio / np.max(np.abs(host_audio)) * 0.7
    
    return host_audio, sample_rate

def test_compression_robustness():
    """Test robustness to compression by simulating compression effects"""
    print("\n" + "="*60)
    print("TESTING COMPRESSION ROBUSTNESS")
    print("="*60)
    
    # Create steganography system optimized for compression robustness
    steg = SpreadSpectrumSteganography(
        chip_rate=1024,      # Higher chip rate for more robustness
        power_ratio=0.003,   # Higher power ratio
        frame_size=2048
    )
    
    # Create host audio and test data
    host_audio, sr = create_host_audio(20.0)
    test_message = "Compression test message " * 20
    test_data = test_message.encode('utf-8')
    
    print(f"Original message length: {len(test_data)} bytes")
    
    # Hide data
    stego_audio = steg.hide_data(host_audio, test_data, sr)
    
    # Simulate compression effects
    def simulate_mp3_compression(audio, quality_factor=0.8):
        """Simulate MP3 compression by applying frequency domain filtering"""
        # Apply slight low-pass filtering and quantization
        fft_audio = np.fft.fft(audio)
        
        # Attenuate high frequencies (MP3 compression effect)
        freqs = np.fft.fftfreq(len(audio))
        high_freq_mask = np.abs(freqs) > 0.3
        fft_audio[high_freq_mask] *= quality_factor
        
        # Quantization effect (reduce precision)
        compressed_audio = np.fft.ifft(fft_audio).real
        compressed_audio = np.round(compressed_audio * 32767) / 32767  # 16-bit quantization
        
        return compressed_audio
    
    # Test different compression levels
    compression_levels = [0.9, 0.8, 0.7, 0.6, 0.5]
    
    for level in compression_levels:
        print(f"\nTesting compression level: {level:.1f}")
        
        # Apply simulated compression
        compressed_stego = simulate_mp3_compression(stego_audio, level)
        
        # Try to extract data
        extracted_data = steg.extract_data(compressed_stego, host_audio, sr)
        
        if extracted_data:
            extracted_message = extracted_data.decode('utf-8', errors='ignore')
            success_rate = len(extracted_message) / len(test_message)
            print(f"  ✅ Extraction successful: {success_rate:.2%} of data recovered")
            
            if extracted_message.strip() == test_message.strip():
                print("  🎉 Perfect recovery!")
            else:
                print(f"  ⚠️  Partial recovery. First 50 chars: {extracted_message[:50]}...")
        else:
            print("  ❌ Extraction failed")

def main():
    """Main demonstration function"""
    print("SPREAD SPECTRUM STEGANOGRAPHY DEMONSTRATION")
    print("="*60)
    
    # Create steganography system
    steg = SpreadSpectrumSteganography(
        chip_rate=512,        # Balance between robustness and capacity
        power_ratio=0.002,    # Low power for imperceptibility
        frame_size=1024,
        overlap=0.5
    )
    
    # Create test files
    print("Creating test files...")
    test_files = create_test_files()
    
    # Create host audio
    print("Creating host audio...")
    host_audio, sample_rate = create_host_audio(duration=25.0)
    host_path = '/tmp/host_audio.wav'
    sf.write(host_path, host_audio, sample_rate)
    print(f"Host audio saved: {host_path}")
    
    # Test hiding different file types
    for file_type, file_path in test_files.items():
        print(f"\n--- Testing {file_type.upper()} file ---")
        
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")
        
        # Hide file
        stego_path = f'/tmp/stego_{file_type}.wav'
        success = steg.hide_file(host_path, file_path, stego_path)
        
        if success:
            print(f"✅ Successfully hidden {file_type} file")
            
            # Extract file
            extracted_path = f'/tmp/extracted_{file_type}.{"txt" if file_type == "text" else "bin"}'
            extract_success = steg.extract_file(stego_path, extracted_path, host_path)
            
            if extract_success:
                extracted_size = os.path.getsize(extracted_path)
                print(f"✅ Successfully extracted: {extracted_size} bytes")
                
                # Verify extraction for text files
                if file_type == 'text':
                    with open(file_path, 'r') as f:
                        original = f.read()
                    with open(extracted_path, 'r') as f:
                        extracted = f.read()
                    
                    if original.strip() == extracted.strip():
                        print("🎉 Perfect text extraction!")
                    else:
                        print("⚠️ Text extraction has some differences")
                
                # Binary comparison
                with open(file_path, 'rb') as f:
                    original_data = f.read()
                with open(extracted_path, 'rb') as f:
                    extracted_data = f.read()
                
                if original_data == extracted_data:
                    print("🎉 Perfect binary match!")
                else:
                    match_ratio = sum(a == b for a, b in zip(original_data, extracted_data)) / len(original_data)
                    print(f"📊 Binary match ratio: {match_ratio:.2%}")
            
            else:
                print("❌ Extraction failed")
        else:
            print(f"❌ Failed to hide {file_type} file")
    
    # Test compression robustness
    test_compression_robustness()
    
    # Clean up test files
    print(f"\n--- Cleaning up ---")
    for file_path in test_files.values():
        try:
            os.unlink(file_path)
        except:
            pass
    
    print("\n🎯 DEMONSTRATION COMPLETE!")
    print("\nGenerated files in /tmp/:")
    for file in os.listdir('/tmp'):
        if file.startswith(('host_', 'stego_', 'extracted_')):
            print(f"  - {file}")

if __name__ == "__main__":
    main()
