# Notes
The code only handles WAV files because the wave module in Python can only read and write uncompressed PCM audio formats like WAV. MP3 is a compressed format and requires special libraries (like pydub, ffmpeg, or librosa) to decode and encode.

Spread spectrum steganography works best on uncompressed audio because compression (like MP3) can distort or remove the hidden data.