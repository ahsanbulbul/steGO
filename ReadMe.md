# steGO - Spread Spectrum Steganography

**Hide any file type inside WAV audio files using spread spectrum techniques**

## Overview

**steGO** is a steganography implementation that can hide any type of file inside WAV audio files using spread spectrum techniques. Unlike traditional LSB steganography, this system distributes hidden data across the frequency spectrum, making it more robust and harder to detect. Also, this is compression resistant (to some extent)

## How It Works

### Spread Spectrum Steganography Theory
- **Data Spreading**: Each bit of hidden data is spread across multiple audio samples using a pseudo-random sequence
- **Chip Rate**: By default, each bit uses 64 "chips" (audio samples) for robustness
- **Low Power Embedding**: Hidden signal uses only 1% of the original audio power
- **Correlation Recovery**: Data is extracted by correlating with the same spreading sequence

### Key Advantages
- **Robust**: More resistant to compression and audio processing
- **Imperceptible**: Hidden data is virtually inaudible
- **Flexible**: Works with any file type
- **Secure**: Uses pseudo-random spreading sequences
- **Metadata Preservation**: Stores filename, type, and checksums

## Onboarding

### Installation
```bash
git clone https://github.com/ahsanbulbul/steGO.git
cd steGO
```

No external dependencies required for basic functionality! Uses only Python standard library.

### Basic Usage

#### Hide a File
```bash
python3 file_steganography.py hide <host_audio.wav> <file_to_hide> [output.wav]
```

#### Extract Hidden File
```bash
python3 file_steganography.py extract <stego_audio.wav> <output_dir> <original_audio.wav>
```

#### Check Capacity
```bash
python3 file_steganography.py capacity <host_audio.wav> [chip_rate]
```

#### File Information
```bash
python3 file_steganography.py info <file_path>
```