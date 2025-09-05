#!/usr/bin/env python3

import numpy as np
import wave
import struct
import hashlib
import os
import sys
import mimetypes
from typing import Tuple, Optional
from spread_spectrum_core import SpreadSpectrumCore

class FileSpreadSpectrumSteganography:
    """
    File Steganography System with Utility Functions
    
    Uses SpreadSpectrumCore for the core algorithms and provides
    file I/O, audio handling, and command-line interface utilities.
    """
    
    def __init__(self, 
                 chip_rate: int = 64,
                 power_ratio: float = 0.01,
                 sample_rate: int = 44100):
        
        self.sample_rate = sample_rate
        
        # Initialize the core spread spectrum engine
        self.core = SpreadSpectrumCore(
            chip_rate=chip_rate,
            power_ratio=power_ratio,
            spreading_seed=162
        )
    
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
                audio_data = (audio_data - 128) / 128.0
            elif sampwidth == 2:
                dtype = np.int16
                audio_data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                audio_data = audio_data / 32768.0
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")
            
            # Convert to mono if stereo
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2)
                audio_data = np.mean(audio_data, axis=1)
            
            return audio_data, sample_rate
    
    def save_wav(self, filename: str, audio_data: np.ndarray, sample_rate: int):
        """Save audio data to WAV file"""
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    def _prepare_file_data(self, file_path: str) -> Tuple[np.ndarray, dict]:
        """Prepare file data with metadata for hiding (wrapper for core)"""
        # Read file data
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Get file metadata
        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
        
        # Use core to prepare data
        return self.core.prepare_file_data(file_data, filename, mime_type)
    
    def _extract_file_data(self, bits: np.ndarray) -> Tuple[bytes, dict]:
        """Extract file data and metadata from bits (wrapper for core)"""
        return self.core.extract_file_data(bits)
    
    def _spread_bits(self, bits: np.ndarray) -> np.ndarray:
        """Spread bits using pseudo-random sequence (wrapper for core)"""
        return self.core.spread_bits(bits)
    
    def _despread_signal(self, signal_chips: np.ndarray) -> np.ndarray:
        """Despread signal to recover bits (wrapper for core)"""
        return self.core.despread_signal(signal_chips)
    
    def calculate_capacity(self, host_audio_path: str) -> dict:
        """Calculate the hiding capacity of the host audio"""
        try:
            host_audio, sr = self.load_wav(host_audio_path)
            total_samples = len(host_audio)
            
            # Use core to calculate capacity
            capacity_bits = self.core.calculate_capacity_bits(total_samples)
            capacity_bytes = max(0, capacity_bits // 8)
            
            return {
                'audio_duration': total_samples / sr,
                'total_samples': total_samples,
                'capacity_bits': capacity_bits,
                'capacity_bytes': capacity_bytes,
                'capacity_kb': capacity_bytes / 1024,
                'chip_rate': self.core.chip_rate
            }
        except Exception as e:
            return {'error': str(e)}
    
    def hide_file(self, host_audio_path: str, 
                  data_file_path: str, 
                  output_path: str) -> bool:
        """
        Hide any type of file in audio
        
        Args:
            host_audio_path: Path to host WAV file
            data_file_path: Path to file to hide (any type)
            output_path: Path for output stego WAV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load host audio
            host_audio, sr = self.load_wav(host_audio_path)
            
            # Read file data
            with open(data_file_path, 'rb') as f:
                file_data = f.read()
            
            # Get file metadata
            filename = os.path.basename(data_file_path)
            mime_type, _ = mimetypes.guess_type(data_file_path)
            if mime_type is None:
                mime_type = "application/octet-stream"
            
            print(f"📁 File: {filename}")
            print(f"🏷️  Type: {mime_type}")
            print(f"📏 Size: {len(file_data):,} bytes")
            print(f"🔧 Hiding file in audio...")
            
            # Use core to hide data
            stego_audio, metadata = self.core.hide_data_in_audio(
                host_audio, file_data, filename, mime_type
            )
            
            # Save stego audio
            self.save_wav(output_path, stego_audio, sr)
            
            print(f"✅ Stego audio saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error hiding file: {e}")
            return False
    
    def extract_file(self, stego_audio_path: str, 
                    output_dir: str,
                    original_audio_path: str) -> bool:
        """
        Extract hidden file from stego audio
        
        Args:
            stego_audio_path: Path to stego WAV file
            output_dir: Directory to save extracted file
            original_audio_path: Path to original WAV file (required)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load stego audio
            stego_audio, sr = self.load_wav(stego_audio_path)
            
            # Load original audio (required)
            original_audio, _ = self.load_wav(original_audio_path)
            
            print("🔍 Extracting hidden file...")
            
            # Use core to extract data
            file_data, metadata = self.core.extract_data_from_audio(stego_audio, original_audio)
            
            if file_data is not None and metadata is not None:
                # Verify checksum
                calculated_checksum = hashlib.md5(file_data).digest()[:4]
                if calculated_checksum == metadata['checksum']:
                    # Save extracted file
                    output_path = os.path.join(output_dir, metadata['filename'])
                    with open(output_path, 'wb') as f:
                        f.write(file_data)
                    
                    print(f"✅ File extracted: {output_path}")
                    print(f"   📁 Filename: {metadata['filename']}")
                    print(f"   🏷️  Type: {metadata['mime_type']}")
                    print(f"   📏 Size: {len(file_data):,} bytes")
                    return True
                else:
                    print("⚠️ Checksum mismatch - file may be corrupted")
                    # Save anyway
                    output_path = os.path.join(output_dir, f"corrupted_{metadata['filename']}")
                    with open(output_path, 'wb') as f:
                        f.write(file_data)
                    print(f"⚠️ File saved (may be corrupted): {output_path}")
                    return True
            else:
                print("❌ Could not extract file")
                return False
                
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return False

def get_file_info(file_path: str):
    """Display file information"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    size = os.path.getsize(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    
    print(f"📁 File: {os.path.basename(file_path)}")
    print(f"📏 Size: {size:,} bytes ({size/1024:.1f} KB)")
    print(f"🏷️  Type: {mime_type}")

def main():
    """Main function with enhanced command line interface"""
    
    if len(sys.argv) < 2:
        print("FILE SPREAD SPECTRUM STEGANOGRAPHY")
        print("Hide ANY file type in WAV audio!")
        print("="*50)
        print()
        print("USAGE:")
        print("  Hide file:    python file_steganography.py hide <host_audio.wav> <file_to_hide> [output.wav]")
        print("  Extract file: python file_steganography.py extract <stego_audio.wav> <output_dir> <original_audio.wav>")
        print("  File info:    python file_steganography.py info <file_path>")
        print("  Capacity:     python file_steganography.py capacity <host_audio.wav> [chip_rate]")
        print()
        print("EXAMPLES:")
        print("  python file_steganography.py hide host.wav secret_image.jpg stego.wav")
        print("  python file_steganography.py hide host.wav document.pdf")
        print("  python file_steganography.py extract stego.wav ./extracted/ host.wav")
        print("  python file_steganography.py capacity host.wav")
        print("  python file_steganography.py info myfile.jpg")
        return
    
    command = sys.argv[1].lower()
    
    if command == "info":
        if len(sys.argv) < 3:
            print("Usage: python file_steganography.py info <file_path>")
            return
        get_file_info(sys.argv[2])
        
    elif command == "capacity":
        if len(sys.argv) < 3:
            print("Usage: python file_steganography.py capacity <host_audio.wav> [chip_rate]")
            return
            
        host_audio_path = sys.argv[2]
        chip_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 64
        
        print("AUDIO CAPACITY ANALYSIS")
        print("="*30)
        
        steg = FileSpreadSpectrumSteganography(chip_rate=chip_rate)
        capacity = steg.calculate_capacity(host_audio_path)
        
        if 'error' in capacity:
            print(f"❌ Error: {capacity['error']}")
        else:
            print(f"🎵 Audio duration: {capacity['audio_duration']:.1f} seconds")
            print(f"📊 Total samples: {capacity['total_samples']:,}")
            print(f"🔧 Chip rate: {capacity['chip_rate']}")
            print(f"📏 Capacity: {capacity['capacity_bytes']:,} bytes ({capacity['capacity_kb']:.1f} KB)")
            
            if capacity['capacity_kb'] > 1024:
                print(f"           {capacity['capacity_kb']/1024:.1f} MB")
        
    elif command == "hide":
        if len(sys.argv) < 4:
            print("Usage: python file_steganography.py hide <host_audio.wav> <file_to_hide> [output.wav]")
            return
            
        host_audio_path = sys.argv[2]
        file_to_hide = sys.argv[3]
        output_path = sys.argv[4] if len(sys.argv) > 4 else "stego_output.wav"
        
        print("HIDING FILE IN AUDIO")
        print("="*30)
        
        # Show file information
        get_file_info(file_to_hide)
        print()
        
        # Check host audio
        if not os.path.exists(host_audio_path):
            print(f"❌ Host audio not found: {host_audio_path}")
            return
        
        try:
            # Calculate optimal chip rate based on file size
            file_size = os.path.getsize(file_to_hide)
            steg = FileSpreadSpectrumSteganography()
            host_audio, sr = steg.load_wav(host_audio_path)
            
            print(f"🎵 Host audio: {len(host_audio):,} samples ({len(host_audio)/sr:.1f}s)")
            
            # Estimate header size (filename + mime + metadata ≈ 100 bytes)
            estimated_total = file_size + 100
            
            # Try different chip rates
            for chip_rate in [16, 32, 64, 128]:
                required_samples = estimated_total * 8 * chip_rate
                if required_samples <= len(host_audio):
                    print(f"✅ Using chip_rate={chip_rate} (capacity check passed)")
                    steg = FileSpreadSpectrumSteganography(chip_rate=chip_rate)
                    break
            else:
                print("⚠️ File might be too large, trying with minimum chip rate...")
                steg = FileSpreadSpectrumSteganography(chip_rate=16)
            
            # Hide the file
            success = steg.hide_file(host_audio_path, file_to_hide, output_path)
            
            if success:
                print(f"\n🎉 SUCCESS! File hidden in: {output_path}")
                print(f"📤 You can now share '{output_path}' - the hidden file is invisible!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif command == "extract":
        if len(sys.argv) < 5:
            print("Usage: python file_steganography.py extract <stego_audio.wav> <output_dir> <original_audio.wav>")
            return
            
        stego_path = sys.argv[2]
        output_dir = sys.argv[3]
        original_path = sys.argv[4]
        
        print("EXTRACTING HIDDEN FILE")
        print("="*30)
        
        if not os.path.exists(stego_path):
            print(f"❌ Stego audio not found: {stego_path}")
            return
        
        if not os.path.exists(original_path):
            print(f"❌ Original audio not found: {original_path}")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Try different chip rates
        for chip_rate in [16, 32, 64, 128, 256]:
            print(f"\n🔍 Trying chip_rate={chip_rate}...")
            
            steg = FileSpreadSpectrumSteganography(chip_rate=chip_rate)
            success = steg.extract_file(stego_path, output_dir, original_path)
            
            if success:
                print(f"\n🎉 SUCCESS! Hidden file extracted to: {output_dir}")
                break
        else:
            print("\n❌ Could not extract file with any chip rate")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'hide', 'extract', 'info', or 'capacity'")

if __name__ == "__main__":
    main()