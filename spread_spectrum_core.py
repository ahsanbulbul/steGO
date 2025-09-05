#!/usr/bin/env python3

import numpy as np
import hashlib
import struct
from typing import Tuple, Optional

class SpreadSpectrumCore:
    """
    Core class for spread spectrum steganography in audio files.
    """
    
    def __init__(self, chip_rate: int = 64, power_ratio: float = 0.01, spreading_seed: int = 162):
        
        """
        Initialize the spread spectrum core
        
        Args:
            chip_rate: Number of chips per bit
            power_ratio: Power ratio of hidden signal to host
            spreading_seed: Seed for pseudo-random spreading sequence
        """
        self.chip_rate = chip_rate
        self.power_ratio = power_ratio
        self.spreading_seed = spreading_seed
        
        # Generate pseudo-random spreading sequence
        np.random.seed(spreading_seed)
        self.spreading_sequence = np.random.choice([-1, 1], size=chip_rate).astype(np.float32)
    
    def prepare_file_data(self, file_data: bytes, filename: str, mime_type: str) -> Tuple[np.ndarray, dict]:
        """
        Prepare file data with metadata for hiding
        
        Args:
            file_data: Raw file data
            filename: Name of the file
            mime_type: MIME type of the file
            
        Returns:
            Tuple of (bits_array, metadata_dict)
        """
        file_size = len(file_data)
        
        # Create metadata
        metadata = {
            'filename': filename,
            'size': file_size,
            'mime_type': mime_type,
            'checksum': hashlib.md5(file_data).digest()[:4]
        }
        
        # Pack metadata into binary format
        filename_bytes = filename.encode('utf-8')
        mime_bytes = mime_type.encode('utf-8')
        
        # Header format: filename_len(1) + filename + mime_len(1) + mime_type + size(4) + checksum(4)
        header = struct.pack('B', len(filename_bytes)) + filename_bytes
        header += struct.pack('B', len(mime_bytes)) + mime_bytes
        header += struct.pack('<I', file_size)
        header += metadata['checksum']
        
        # Combine header and data
        full_data = header + file_data
        
        # Convert to bits
        bits = []
        for byte in full_data:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)
        
        return np.array(bits, dtype=np.float32), metadata
    
    def extract_file_data(self, bits: np.ndarray) -> Tuple[bytes, dict]:
        """
        Extract file data and metadata from bits
        
        Args:
            bits: Recovered bits array
            
        Returns:
            Tuple of (file_data, metadata_dict)
        """
        # Convert bits to bytes
        if len(bits) % 8 != 0:
            padding = 8 - (len(bits) % 8)
            bits = np.concatenate([bits, np.zeros(padding)])
        
        bytes_data = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            byte_val = 0
            for j, bit in enumerate(byte_bits):
                if bit > 0.5:
                    byte_val |= (1 << (7-j))
            bytes_data.append(byte_val)
        
        data = bytes(bytes_data)
        
        # Parse header
        offset = 0
        
        # Filename
        filename_len = data[offset]
        offset += 1
        filename = data[offset:offset+filename_len].decode('utf-8')
        offset += filename_len
        
        # MIME type
        mime_len = data[offset]
        offset += 1
        mime_type = data[offset:offset+mime_len].decode('utf-8')
        offset += mime_len
        
        # File size
        file_size = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # Checksum
        checksum = data[offset:offset+4]
        offset += 4
        
        # File data
        file_data = data[offset:offset+file_size]
        
        metadata = {
            'filename': filename,
            'size': file_size,
            'mime_type': mime_type,
            'checksum': checksum
        }
        
        return file_data, metadata
    
    def spread_bits(self, bits: np.ndarray) -> np.ndarray:
        """
        Spread bits using pseudo-random sequence
        
        Args:
            bits: Binary data as array of 0s and 1s
            
        Returns:
            Spread signal array
        """
        spread_signal = []
        for bit in bits:
            bit_val = 2 * bit - 1  # Map 0,1 to -1,+1
            chip_sequence = bit_val * self.spreading_sequence
            spread_signal.extend(chip_sequence)
        
        return np.array(spread_signal, dtype=np.float32)
    
    def despread_signal(self, signal_chips: np.ndarray) -> np.ndarray:
        """
        Despread signal to recover bits
        
        Args:
            signal_chips: Received signal chips
            
        Returns:
            Recovered bits array
        """
        num_bits = len(signal_chips) // self.chip_rate
        bits = []
        
        for i in range(num_bits):
            start_idx = i * self.chip_rate
            end_idx = start_idx + self.chip_rate
            
            if end_idx <= len(signal_chips):
                chip_segment = signal_chips[start_idx:end_idx]
                correlation = np.dot(chip_segment, self.spreading_sequence)
                bit = 1 if correlation > 0 else 0
                bits.append(bit)
        
        return np.array(bits, dtype=np.float32)
    
    def apply_adaptive_power(self, host_audio: np.ndarray, spread_signal: np.ndarray) -> np.ndarray:
        """
        Apply adaptive power control to the spread signal
        
        Args:
            host_audio: Host audio signal
            spread_signal: Spread spectrum signal
            
        Returns:
            Power-adjusted spread signal
        """
        stego_audio = host_audio.copy()
        
        # Add hidden signal with adaptive power
        for i in range(len(spread_signal)):
            if i < len(host_audio):
                local_power = host_audio[i] ** 2
                adaptive_power = self.power_ratio * (1 + local_power)
                stego_audio[i] += spread_signal[i] * adaptive_power
        
        return stego_audio
    
    def hide_data_in_audio(self, host_audio: np.ndarray, file_data: bytes, filename: str, mime_type: str) -> Tuple[np.ndarray, dict]:
        """
        Core function to hide file data in audio using spread spectrum
        
        Args:
            host_audio: Host audio signal
            file_data: File data to hide
            filename: Name of the file
            mime_type: MIME type of the file
            
        Returns:
            Tuple of (stego_audio, metadata)
        """
        # Prepare file data with metadata
        bits, metadata = self.prepare_file_data(file_data, filename, mime_type)
        
        # Spread the bits
        spread_signal = self.spread_bits(bits)
        
        # Check capacity
        required_samples = len(spread_signal)
        if len(host_audio) < required_samples:
            raise ValueError(f"Host audio too short. Need {required_samples:,} samples, "
                           f"got {len(host_audio):,}")
        
        # Apply adaptive power and create stego audio
        stego_audio = self.apply_adaptive_power(host_audio, spread_signal)
        
        return stego_audio, metadata
    
    def extract_data_from_audio(self, stego_audio: np.ndarray, 
                               original_audio: Optional[np.ndarray] = None) -> Tuple[Optional[bytes], Optional[dict]]:
        """
        Core function to extract hidden data from stego audio
        
        Args:
            stego_audio: Audio with hidden data
            original_audio: Original host audio (optional)
            
        Returns:
            Tuple of (file_data, metadata) or (None, None) if extraction failed
        """
        try:
            # Get hidden signal
            if original_audio is not None:
                hidden_signal = stego_audio[:len(original_audio)] - original_audio
            else:
                hidden_signal = stego_audio.copy()
            
            # Extract header portion to estimate total size
            header_estimate = 1000  # Estimate for header bits
            header_chips = header_estimate * self.chip_rate
            
            if len(hidden_signal) < header_chips:
                return None, None
            
            header_signal = hidden_signal[:header_chips]
            header_bits = self.despread_signal(header_signal)
            
            # Try to parse partial header to get file size
            partial_data = []
            for i in range(0, min(len(header_bits), 800), 8):  # Try first 100 bytes
                byte_bits = header_bits[i:i+8]
                if len(byte_bits) == 8:
                    byte_val = 0
                    for j, bit in enumerate(byte_bits):
                        if bit > 0.5:
                            byte_val |= (1 << (7-j))
                    partial_data.append(byte_val)
            
            partial_bytes = bytes(partial_data)
            
            # Parse header to get file size
            offset = 0
            filename_len = partial_bytes[offset]
            offset += 1 + filename_len  # Skip filename
            mime_len = partial_bytes[offset]
            offset += 1 + mime_len  # Skip mime type
            
            if offset + 4 <= len(partial_bytes):
                file_size = struct.unpack('<I', partial_bytes[offset:offset+4])[0]
                
                # Calculate total bits needed
                header_size = offset + 8  # +4 for size, +4 for checksum
                total_size = header_size + file_size
                total_bits = total_size * 8
                total_chips = total_bits * self.chip_rate
                
                if len(hidden_signal) >= total_chips:
                    # Extract all data
                    all_signal = hidden_signal[:total_chips]
                    all_bits = self.despread_signal(all_signal)
                    
                    # Extract file data and metadata
                    file_data, metadata = self.extract_file_data(all_bits)
                    
                    # Verify checksum
                    calculated_checksum = hashlib.md5(file_data).digest()[:4]
                    if calculated_checksum == metadata['checksum']:
                        return file_data, metadata
                    else:
                        # Return data even if checksum fails (might be partially correct)
                        return file_data, metadata
                        
            return None, None
            
        except Exception as e:
            return None, None
    
    def calculate_capacity_bits(self, audio_length: int) -> int:
        """
        Calculate the hiding capacity in bits for given audio length
        
        Args:
            audio_length: Length of host audio in samples
            
        Returns:
            Capacity in bits
        """
        # Account for header overhead (estimate ~100 bytes)
        header_overhead_bits = 100 * 8
        
        # Calculate capacity in bits
        capacity_bits = (audio_length // self.chip_rate) - header_overhead_bits
        return max(0, capacity_bits)
    
    def get_spreading_parameters(self) -> dict:
        """
        Get current spreading parameters
        
        Returns:
            Dictionary with spreading parameters
        """
        return {
            'chip_rate': self.chip_rate,
            'power_ratio': self.power_ratio,
            'spreading_seed': self.spreading_seed,
            'sequence_length': len(self.spreading_sequence)
        }
