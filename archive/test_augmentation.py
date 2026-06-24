"""
Test script to demonstrate audio augmentation with a single file
"""

import os
import librosa
from audio_augmentation import augment_audio_file

def test_augmentation():
    """Test audio augmentation with a single file"""
    
    # Check if we have any audio files to work with
    pd_dir = r"c:\Users\ashut\Downloads\4th Year\Speech_Healthy_PD[1]\Speech_Healthy_PD[1]\23849127\PD_AH\PD_AH"
    
    # Look for audio files in the PD directory
    if os.path.exists(pd_dir):
        pd_files = [f for f in os.listdir(pd_dir) if f.endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac', '.WAV'))]
        if pd_files:
            # Use the first file for testing
            input_file = os.path.join(pd_dir, pd_files[0])
            output_file = r"c:\Users\ashut\Downloads\4th Year\Speech_Healthy_PD[1]\Speech_Healthy_PD[1]\23849127\test_augmented.wav"
            
            print(f"Testing augmentation on: {input_file}")
            print(f"Output will be saved to: {output_file}")
            
            # Test with specific techniques
            techniques = ['noise', 'pitch_shift']
            print(f"Applying techniques: {techniques}")
            
            try:
                augment_audio_file(input_file, output_file, techniques)
                print("Augmentation completed successfully!")
            except Exception as e:
                print(f"Error during augmentation: {e}")
        else:
            print("No audio files found in PD_AH/PD_AH directory.")
    else:
        print("PD_AH/PD_AH directory not found.")

if __name__ == "__main__":
    test_augmentation()