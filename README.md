# Parkinson's Disease Detection Using Speech Analysis

This project uses machine learning techniques to detect Parkinson's Disease (PD) from speech samples. The approach leverages audio feature extraction, feature selection using multiple optimization algorithms, and classification with Random Forest.

## Project Overview

Parkinson's Disease is a neurodegenerative disorder that affects movement and speech. Early detection is crucial for effective treatment. This project demonstrates how speech analysis can be used as a non-invasive method for PD detection.

## Dataset

The dataset contains 81 audio samples:
- 40 samples from Parkinson's patients (PwPD)
- 41 samples from healthy controls (HC)

Audio files are stored in:
- [HC_AH/HC_AH](file:///c%3A/Users/ashut/Downloads/4th%20Year/Speech_Healthy_PD%5B1%5D/Speech_Healthy_PD%5B1%5D/23849127/HC_AH/HC_AH) - Healthy control samples
- [PD_AH/PD_AH](file:///c%3A/Users/ashut/Downloads/4th%20Year/Speech_Healthy_PD%5B1%5D/Speech_Healthy_PD%5B1%5D/23849127/PD_AH/PD_AH) - Parkinson's patient samples

Demographics information is available in [Demographics_age_sex.xlsx](file:///c%3A/Users/ashut/Downloads/4th%20Year/Speech_Healthy_PD%5B1%5D/Speech_Healthy_PD%5B1%5D/23849127/Demographics_age_sex.xlsx)

## Methodology

1. **Feature Extraction**: Using Librosa library to extract 43 audio features:
   - MFCC (13 mean, 13 std)
   - Spectral features (centroid, rolloff, bandwidth)
   - Zero Crossing Rate (ZCR)
   - Root Mean Square (RMS)
   - Chroma features (12 dimensions)

2. **Feature Selection**: Multiple optimization algorithms to select the most discriminative features:
   - Gray Wolf Optimization (GWO)
   - Artificial Bee Colony (ABC)
   - Particle Swarm Optimization (PSO)

3. **Classification**: Random Forest classifier to distinguish between PD patients and healthy controls

## Key Results

- All algorithms selected 13 optimal features out of 43 (69.77% reduction)
- Accuracy of 76% with selected features
- Demonstrates that feature selection is crucial for medical applications

## Requirements

- Python 3.x
- Librosa
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit
- SoundFile
- SciPy

## Usage

1. Ensure all audio files are in the correct directory structure
2. Run the Jupyter notebook [Parkinsons_Detection.ipynb](file:///c%3A/Users/ashut/Downloads/4th%20Year/Speech_Healthy_PD%5B1%5D/Speech_Healthy_PD%5B1%5D/23849127/Parkinsons_Detection.ipynb)
3. Alternatively, run the Python script [parkinsons_detection.py](file:///c%3A/Users/ashut/Downloads/4th%20Year/Speech_Healthy_PD%5B1%5D/Speech_Healthy_PD%5B1%5D/23849127/parkinsons_detection.py)
4. The program will automatically process the audio files and compare all three algorithms

## Audio Data Augmentation

To generate new samples from existing audio files using data augmentation techniques:

```bash
python audio_augmentation.py --input_dir "PD_AH/PD_AH" --output_dir "PD_AH_Augmented" --samples_per_file 5
```

This will generate 5 augmented samples for each original file in the PD_AH directory. Available augmentation techniques include:
- Noise addition
- Time stretching
- Pitch shifting
- Time shifting
- Speed changing
- Volume changing
- Low-pass filtering
- High-pass filtering

## Web Application

To run the Streamlit web application for visualizing results:

```bash
streamlit run parkinsons_app.py
```

This will launch a web interface showing:
- Algorithm comparison
- Feature analysis
- Model performance metrics
- Detailed classification reports

## Key Insights

1. Feature selection is critical for performance in medical applications
2. Too many features can hurt accuracy due to noise
3. Multiple optimization algorithms can achieve similar performance
4. Medical applications benefit from simpler, interpretable models

## Future Work

- Implement additional feature selection algorithms for comparison
- Try different classification algorithms
- Add visualization of audio features
- Expand the dataset with more samples
- Implement cross-validation for more robust evaluation
- Use advanced generative models for sample generation