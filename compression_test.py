import os
import subprocess
from file_steganography import FileSpreadSpectrumSteganography


def compare_files(original_path, extracted_path):
    """Return percentage of matching bytes between two files"""
    with open(original_path, "rb") as f1, open(extracted_path, "rb") as f2:
        orig = f1.read()
        extr = f2.read()
    min_len = min(len(orig), len(extr))
    if min_len == 0:
        return 0
    matches = sum(a == b for a, b in zip(orig[:min_len], extr[:min_len]))
    percent = matches / min_len * 100
    return percent


def compress_and_test(stego_wav, original_wav, original_file, output_dir, bitrates):
    print("COMPRESSION & EXTRACTION TEST")
    print("=" * 40)
    for br in bitrates:
        mp3_path = os.path.join(output_dir, f"stego_{br}kbps.mp3")
        recon_wav = os.path.join(output_dir, f"recon_{br}kbps.wav")
        # Convert WAV to MP3
        subprocess.run(
            ["ffmpeg", "-y", "-i", stego_wav, "-b:a", f"{br}k", mp3_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Convert MP3 back to WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, recon_wav],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Calculate compression ratio
        orig_size = os.path.getsize(stego_wav)
        mp3_size = os.path.getsize(mp3_path)
        ratio = mp3_size / orig_size * 100
        print(f"\nBitrate: {br} kbps | Compression: {ratio:.1f}%")
        # Try extraction
        steg = FileSpreadSpectrumSteganography()
        success = steg.extract_file(recon_wav, output_dir, original_wav)
        if success:
            # Find extracted file path
            extracted_files = [
                f
                for f in os.listdir(output_dir)
                if f != os.path.basename(original_file)
            ]
            if extracted_files:
                extracted_path = os.path.join(output_dir, extracted_files[0])
                percent = compare_files(original_file, extracted_path)
                print(f"Extractable: {percent:.1f}% of bytes match")
            else:
                print("Extractable: Could not find extracted file")
        else:
            print("Extractable: Extraction failed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python compression_test.py <original.wav> <file_to_hide> <output_dir>"
        )
        sys.exit(1)
    original_wav = sys.argv[1]
    file_to_hide = sys.argv[2]
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)
    stego_wav = os.path.join(output_dir, "stego.wav")

    # Hide the file in the WAV
    steg = FileSpreadSpectrumSteganography()
    success = steg.hide_file(original_wav, file_to_hide, stego_wav)
    if not success:
        print("Failed to hide file in audio.")
        sys.exit(1)

    bitrates = [320, 192, 128, 96, 64]
    compress_and_test(stego_wav, original_wav, file_to_hide, output_dir, bitrates)
