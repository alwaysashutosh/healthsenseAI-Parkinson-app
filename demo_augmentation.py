"""
Demo script to show how to use the audio augmentation techniques
"""

import os
import random
from audio_augmentation import augment_audio_file

def demo_augmentation():
    """Demonstrate audio augmentation techniques"""
    
    # Check if we have any audio files to work with
    pd_dir = "PD_AH/PD_AH"
    hc_dir = "HC_AH/HC_AH"
    
    # Look for audio files in both directories
    audio_files = []
    
    if os.path.exists(pd_dir):
        pd_files = [f for f in os.listdir(pd_dir) if f.endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac', '.WAV'))]
        if pd_files:
            audio_files.append(os.path.join(pd_dir, pd_files[0]))
    
    if os.path.exists(hc_dir):
        hc_files = [f for f in os.listdir(hc_dir) if f.endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac', '.WAV'))]
        if hc_files:
            audio_files.append(os.path.join(hc_dir, hc_files[0]))
    
    if not audio_files:
        print("No audio files found. Please ensure you have audio files in PD_AH/PD_AH or HC_AH/HC_AH directories.")
        return
    
    # Create output directory
    output_dir = "demo_augmented"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Audio Data Augmentation Demo")
    print("=" * 30)
    
    # Demonstrate different augmentation techniques
    techniques_list = [
        ['noise'],
        ['pitch_shift'],
        ['time_stretch'],
        ['volume'],
        ['low_pass'],
        ['high_pass'],
        ['noise', 'pitch_shift'],
        ['time_stretch', 'volume'],
        ['low_pass', 'high_pass']
    ]
    
    for i, input_file in enumerate(audio_files[:2]):  # Process max 2 files
        print(f"\nProcessing: {input_file}")
        
        # Get base filename without path and extension
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        for j, techniques in enumerate(techniques_list):
            output_file = os.path.join(output_dir, f"{base_name}_aug_{j+1}.wav")
            print(f"  Generating augmented sample with techniques: {techniques}")
            try:
                augment_audio_file(input_file, output_file, techniques)
            except Exception as e:
                print(f"    Error: {e}")
    
    print(f"\nDemo completed. Check the '{output_dir}' directory for augmented samples.")

if __name__ == "__main__":
    demo_augmentation()