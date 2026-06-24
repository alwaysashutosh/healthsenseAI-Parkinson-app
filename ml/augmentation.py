import librosa
import numpy as np
import soundfile as sf
import os
import random
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import argparse
from typing import Tuple, Any

def add_noise(data, noise_factor=0.03):
    """Add random noise to the audio signal"""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to original data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def time_stretch(data, rate=1.1):
    """Time stretching audio without changing pitch"""
    return librosa.effects.time_stretch(y=data, rate=rate)

def pitch_shift(data, sr, n_steps=2):
    """Pitch shifting audio without changing tempo"""
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=int(n_steps))

def time_shift(data, shift_factor=0.1):
    """Shift audio in time"""
    shift = int(len(data) * shift_factor)
    if shift > 0:
        # Shift to the right
        augmented_data = np.concatenate([np.zeros(shift), data[:-shift]])
    else:
        # Shift to the left
        augmented_data = np.concatenate([data[-shift:], np.zeros(-shift)])
    return augmented_data

def change_speed(data, speed_factor=1.1):
    """Change the speed of the audio"""
    # Resample to change speed
    indices = np.round(np.arange(0, len(data), speed_factor)).astype(int)
    indices = indices[indices < len(data)]
    return data[indices]

def change_volume(data, factor=1.2):
    """Change the volume of the audio"""
    return data * factor

def apply_low_pass_filter(data, sr, cutoff=3000):
    """Apply a low-pass filter to simulate muffled speech"""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    # Fix the tuple unpacking issue
    filter_coeffs = butter(5, normal_cutoff, btype='low', analog=False)
    if isinstance(filter_coeffs, tuple) and len(filter_coeffs) >= 2:
        b, a = filter_coeffs[0], filter_coeffs[1]
    else:
        # Handle case where filter_coeffs might be a single array or other structure
        b = filter_coeffs if hasattr(filter_coeffs, '__len__') else np.array([1.0])
        a = np.array([1.0])
    return lfilter(b, a, data)

def apply_high_pass_filter(data, sr, cutoff=100):
    """Apply a high-pass filter to simulate breathy speech"""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    # Fix the tuple unpacking issue
    filter_coeffs = butter(5, normal_cutoff, btype='high', analog=False)
    if isinstance(filter_coeffs, tuple) and len(filter_coeffs) >= 2:
        b, a = filter_coeffs[0], filter_coeffs[1]
    else:
        # Handle case where filter_coeffs might be a single array or other structure
        b = filter_coeffs if hasattr(filter_coeffs, '__len__') else np.array([1.0])
        a = np.array([1.0])
    return lfilter(b, a, data)

def augment_audio_file(input_path, output_path, techniques=None):
    """
    Apply various augmentation techniques to an audio file
    
    Parameters:
    input_path (str): Path to the input audio file
    output_path (str): Path to save the augmented audio file
    techniques (list): List of techniques to apply. If None, randomly select some.
    """
    # Load the audio file
    data, sr = librosa.load(path=input_path, sr=None)
    
    # If no techniques specified, randomly select some
    if techniques is None:
        all_techniques = [
            'noise', 'time_stretch', 'pitch_shift', 
            'time_shift', 'speed', 'volume', 
            'low_pass', 'high_pass'
        ]
        # Randomly select 2-4 techniques
        techniques = random.sample(all_techniques, random.randint(2, 4))
    
    # Apply selected techniques
    augmented_data = data.copy()
    
    for technique in techniques:
        if technique == 'noise':
            augmented_data = add_noise(augmented_data)
        elif technique == 'time_stretch':
            rate = random.uniform(0.9, 1.1)
            augmented_data = time_stretch(augmented_data, rate=rate)
        elif technique == 'pitch_shift':
            n_steps = random.uniform(-2, 2)
            augmented_data = pitch_shift(augmented_data, sr, n_steps=int(n_steps))
        elif technique == 'time_shift':
            shift_factor = random.uniform(-0.1, 0.1)
            augmented_data = time_shift(augmented_data, shift_factor=shift_factor)
        elif technique == 'speed':
            speed_factor = random.uniform(0.9, 1.1)
            augmented_data = change_speed(augmented_data, speed_factor=speed_factor)
        elif technique == 'volume':
            factor = random.uniform(0.8, 1.2)
            augmented_data = change_volume(augmented_data, factor=factor)
        elif technique == 'low_pass':
            cutoff = random.randint(2000, 4000)
            augmented_data = apply_low_pass_filter(augmented_data, sr, cutoff=cutoff)
        elif technique == 'high_pass':
            cutoff = random.randint(50, 200)
            augmented_data = apply_high_pass_filter(augmented_data, sr, cutoff=cutoff)
    
    # Save the augmented audio
    sf.write(output_path, augmented_data, sr)
    print(f"Augmented audio saved to: {output_path}")
    print(f"Applied techniques: {techniques}")

def generate_multiple_samples(input_dir, output_dir, samples_per_file=3):
    """
    Generate multiple augmented samples from each file in a directory
    
    Parameters:
    input_dir (str): Directory containing original audio files
    output_dir (str): Directory to save augmented audio files
    samples_per_file (int): Number of augmented samples to generate per original file
    """
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        print(f"Error: Permission denied when creating directory '{output_dir}'.")
        print("Try running the script with administrator privileges or use a different output directory.")
        return
    except Exception as e:
        print(f"Error: Failed to create directory '{output_dir}': {e}")
        return
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(input_dir) 
                   if f.endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac', '.WAV'))]
    
    if not audio_files:
        print(f"No audio files found in '{input_dir}'.")
        return
    
    print(f"Found {len(audio_files)} audio files in {input_dir}")
    
    # Generate augmented samples
    for i, filename in enumerate(audio_files):
        input_path = os.path.join(input_dir, filename)
        print(f"Processing file {i+1}/{len(audio_files)}: {filename}")
        
        for j in range(samples_per_file):
            output_filename = f"aug_{j+1}_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            try:
                augment_audio_file(input_path, output_path)
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
    
    print(f"Generated {len(audio_files) * samples_per_file} augmented samples")

def main():
    parser = argparse.ArgumentParser(description="Audio Data Augmentation for Parkinson's Disease Detection")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing original audio files")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save augmented audio files")
    parser.add_argument("--samples_per_file", type=int, default=3, 
                        help="Number of augmented samples to generate per original file")
    
    args = parser.parse_args()
    
    generate_multiple_samples(args.input_dir, args.output_dir, args.samples_per_file)

if __name__ == "__main__":
    # Check if script is run with command line arguments
    import sys
    if len(sys.argv) > 1:
        # Run with command line arguments
        main()
    else:
        # Show help information
        print("Available augmentation techniques:")
        print("1. Noise addition")
        print("2. Time stretching")
        print("3. Pitch shifting")
        print("4. Time shifting")
        print("5. Speed changing")
        print("6. Volume changing")
        print("7. Low-pass filtering")
        print("8. High-pass filtering")
        print("\nTo use this script, run:")
        print("python audio_augmentation.py --input_dir 'PD_AH/PD_AH' --output_dir 'PD_AH_Augmented' --samples_per_file 5")