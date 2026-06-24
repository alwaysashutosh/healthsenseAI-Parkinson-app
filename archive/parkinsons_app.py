import streamlit as st
import numpy as np
import pandas as pd

# Handle matplotlib import error gracefully
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except (FileNotFoundError, ImportError) as e:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib not available. Visualizations will be disabled.")

from sklearn.metrics import confusion_matrix
import librosa

import os
from io import BytesIO
import base64
import queue
import threading
import time
from collections import deque
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Import backend logic
from parkinsons_detection import LivePredictor

# Set page configuration
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    .stAlert {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2980b9;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏥 Parkinson's Disease Detection Using Speech Analysis")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Project Overview", 
    "Real-Time Monitor",
    "Algorithm Comparison", 
    "Feature Analysis", 
    "Model Performance", 
    "About"
])

# Sample data for demonstration (in a real app, this would come from actual model results)
# For demonstration purposes, I'll create sample data that mimics real results
np.random.seed(42)

# Sample algorithm results
algorithm_results = {
    "Gray Wolf Optimization": {
        "accuracy": 0.76,
        "precision_healthy": 0.71,
        "recall_healthy": 0.92,
        "precision_pd": 0.88,
        "recall_pd": 0.58,
        "features_selected": 13,
        "confusion_matrix": np.array([[12, 1], [5, 7]]),
        "selected_features": [
            "MFCC1_mean", "MFCC2_mean", "MFCC3_mean", "MFCC4_mean", 
            "MFCC6_mean", "MFCC8_mean", "MFCC1_std", "MFCC2_std", 
            "MFCC3_std", "MFCC7_std", "MFCC10_std", "MFCC12_std", "Chroma8"
        ],
        "feature_importance": [
            ("MFCC2_mean", 0.1790), ("MFCC10_std", 0.1048), ("MFCC2_std", 0.0961),
            ("MFCC1_std", 0.0937), ("MFCC12_std", 0.0915), ("MFCC3_std", 0.0741),
            ("MFCC4_mean", 0.0570), ("MFCC7_std", 0.0535), ("MFCC1_mean", 0.0541),
            ("MFCC8_mean", 0.0438), ("MFCC3_mean", 0.0475), ("MFCC6_mean", 0.0351),
            ("Chroma8", 0.0699)
        ]
    },
    "Artificial Bee Colony": {
        "accuracy": 0.72,
        "precision_healthy": 0.69,
        "recall_healthy": 0.85,
        "precision_pd": 0.82,
        "recall_pd": 0.50,
        "features_selected": 15,
        "confusion_matrix": np.array([[11, 2], [5, 7]]),
        "selected_features": [
            "MFCC1_mean", "MFCC2_mean", "MFCC3_mean", "MFCC5_mean",
            "MFCC6_mean", "MFCC7_mean", "MFCC1_std", "MFCC2_std",
            "MFCC4_std", "MFCC7_std", "MFCC9_std", "MFCC10_std",
            "MFCC11_std", "Chroma7", "Chroma8"
        ],
        "feature_importance": [
            ("MFCC2_mean", 0.1620), ("MFCC10_std", 0.1105), ("MFCC2_std", 0.0891),
            ("MFCC1_std", 0.0877), ("MFCC11_std", 0.0815), ("MFCC4_std", 0.0721),
            ("MFCC3_mean", 0.0650), ("MFCC7_std", 0.0585), ("MFCC1_mean", 0.0521),
            ("MFCC7_mean", 0.0498), ("MFCC9_std", 0.0475), ("MFCC5_mean", 0.0391),
            ("Chroma8", 0.0542), ("Chroma7", 0.0415), ("MFCC6_mean", 0.0293)
        ]
    },
    "Particle Swarm Optimization": {
        "accuracy": 0.74,
        "precision_healthy": 0.70,
        "recall_healthy": 0.88,
        "precision_pd": 0.85,
        "recall_pd": 0.53,
        "features_selected": 14,
        "confusion_matrix": np.array([[12, 1], [6, 6]]),
        "selected_features": [
            "MFCC1_mean", "MFCC2_mean", "MFCC3_mean", "MFCC4_mean",
            "MFCC6_mean", "MFCC8_mean", "MFCC1_std", "MFCC2_std",
            "MFCC3_std", "MFCC6_std", "MFCC10_std", "MFCC12_std",
            "Chroma6", "Chroma8"
        ],
        "feature_importance": [
            ("MFCC2_mean", 0.1710), ("MFCC10_std", 0.1078), ("MFCC2_std", 0.0921),
            ("MFCC1_std", 0.0907), ("MFCC12_std", 0.0865), ("MFCC3_std", 0.0711),
            ("MFCC4_mean", 0.0560), ("MFCC6_std", 0.0545), ("MFCC1_mean", 0.0511),
            ("MFCC8_mean", 0.0428), ("MFCC3_mean", 0.0455), ("MFCC6_mean", 0.0361),
            ("Chroma8", 0.0682), ("Chroma6", 0.0625)
        ]
    }
}

# Feature names for display
all_features = [
    'MFCC1_mean', 'MFCC2_mean', 'MFCC3_mean', 'MFCC4_mean', 'MFCC5_mean', 'MFCC6_mean', 
    'MFCC7_mean', 'MFCC8_mean', 'MFCC9_mean', 'MFCC10_mean', 'MFCC11_mean', 'MFCC12_mean', 'MFCC13_mean',
    'MFCC1_std', 'MFCC2_std', 'MFCC3_std', 'MFCC4_std', 'MFCC5_std', 'MFCC6_std',
    'MFCC7_std', 'MFCC8_std', 'MFCC9_std', 'MFCC10_std', 'MFCC11_std', 'MFCC12_std', 'MFCC13_std',
    'Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Bandwidth',
    'ZCR', 'RMS',
    'Chroma1', 'Chroma2', 'Chroma3', 'Chroma4', 'Chroma5', 'Chroma6',
    'Chroma7', 'Chroma8', 'Chroma9', 'Chroma10', 'Chroma11', 'Chroma12'
]

# Page content based on selection
if page == "Project Overview":
    st.header("Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This application demonstrates Parkinson's Disease detection using speech analysis with multiple 
        optimization algorithms for feature selection.
        
        ### How It Works
        1. **Audio Feature Extraction**: 43 features extracted from speech samples using Librosa
        2. **Feature Selection**: Multiple optimization algorithms select the most discriminative features
        3. **Classification**: Random Forest classifier distinguishes between PD patients and healthy controls
        
        ### Dataset
        - 81 audio samples (40 Parkinson's patients, 41 healthy controls)
        - Audio features include MFCC, spectral features, and chroma features
        """)
    
    with col2:
        st.metric("Total Samples", "81")
        st.metric("Parkinson's Samples", "40")
        st.metric("Healthy Samples", "41")
        st.metric("Features Extracted", "43")
    
    st.markdown("---")
    
    st.subheader("Key Results")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">76%</div><div class="metric-label">Best Accuracy</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">69.8%</div><div class="metric-label">Feature Reduction</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Algorithms</div></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">13</div><div class="metric-label">Features Selected</div></div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">13</div><div class="metric-label">Features Selected</div></div>', unsafe_allow_html=True)

# === REALTIME MODULE START ===
elif page == "Real-Time Monitor":
    st.header("🧠 Real-Time Voice Monitoring")
    st.markdown("---")
    
    # Initialize session state for real-time data
    if 'probability_history' not in st.session_state:
        st.session_state.probability_history = []
        st.session_state.time_history = []
    
    if 'prediction_log' not in st.session_state:
        st.session_state.prediction_log = deque(maxlen=10)
        
    # Sidebar controls
    with st.sidebar:
        st.subheader("Monitor Settings")
        algo_selection = st.selectbox("Feature Selection Algorithm", ["Gray Wolf Optimization (Default)", "Artificial Bee Colony", "Particle Swarm Optimization"])
        confidence_threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
        
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Microphone Input")
        
        # Audio callback for WebRTC
        # We use a queue to transfer frames from the WebRTC thread to the Streamlit script
        if 'audio_queue' not in st.session_state:
            st.session_state.audio_queue = queue.Queue()
            
        def audio_frame_callback(frame):
            sound = frame.to_ndarray()
            # Resample or just pass raw if samplerate matches (assuming standard 48k or 44.1k input, will need resampling in Predictor/Extraction)
            # For simplicity in this demo, we assume the backend handles resampling or we get compatible audio
            # But av usually gives 48kHz stereo. We need to convert to mono and maybe queue it.
            
            # Convert to mono and float32
            if sound.ndim > 1:
                sound = np.mean(sound, axis=1)
            
            # Normalize to [-1, 1] if int16
            if sound.dtype == np.int16:
                sound = sound.astype(np.float32) / 32768.0
                
            st.session_state.audio_queue.put(sound)
            return frame

        # WebRTC Streamer
        ctx = webrtc_streamer(
            key="parkinsons-live",
            mode=WebRtcMode.SENDONLY,
            audio_frame_callback=audio_frame_callback,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": False, "audio": True},
        )
        
        st.markdown("### Live Status")
        status_placeholder = st.empty()
        
        if ctx.state.playing:
            status_placeholder.markdown("🔴 **Listening...**")
        else:
            status_placeholder.markdown("⚪ **Idle**")
            
    with col2:
        st.subheader("Real-Time Analysis")
        
        # UI Components
        gauge_placeholder = st.empty()
        chart_placeholder = st.empty()
        log_placeholder = st.empty()
        
        # Initialize Predictor (Load models once)
        if 'live_predictor' not in st.session_state:
            with st.spinner("Loading AI Models..."):
                # Ensure model exists
                if not os.path.exists('rf_model.pkl'):
                    st.error("Model artifacts not found! Please run the training script first or wait for it to complete.")
                else:
                    st.session_state.live_predictor = LivePredictor()
        
        # Processing Loop
        if ctx.state.playing and 'live_predictor' in st.session_state and st.session_state.live_predictor.model is not None:
            
            # Consuming queue
            while not st.session_state.audio_queue.empty():
                audio_chunk = st.session_state.audio_queue.get()
                st.session_state.live_predictor.process_audio_chunk(audio_chunk)
            
            # Predict
            result = st.session_state.live_predictor.predict_live()
            
            if result:
                if result.get('is_silence'):
                    status_placeholder.markdown("🟡 **Silence Detected**")
                else:
                    status_placeholder.markdown("🟢 **Processing Voice**")
                    
                    label = result['label']
                    prob = result['pd_probability']
                    
                    # Update History
                    current_time = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.probability_history.append(prob)
                    st.session_state.time_history.append(current_time)
                    
                    if len(st.session_state.probability_history) > 50:
                        st.session_state.probability_history.pop(0)
                        st.session_state.time_history.pop(0)
                    
                    # Log
                    log_entry = f"[{current_time}] {label} ({prob:.1%})"
                    st.session_state.prediction_log.appendleft(log_entry)
                    
                    # Update Gauge
                    color = "red" if prob > confidence_threshold else "green"
                    gauge_html = f"""
                    <div style="text-align: center;">
                        <span style="font-size: 1.5rem; color: #7f8c8d;">Parkinson's Probability</span>
                        <div style="background-color: #ecf0f1; border-radius: 10px; padding: 2px;">
                            <div style="width: {prob*100}%; background-color: {color}; height: 24px; border-radius: 8px; transition: width 0.5s;"></div>
                        </div>
                        <span style="font-size: 2.5rem; font-weight: bold; color: {color};">{prob:.1%}</span>
                    </div>
                    """
                    gauge_placeholder.markdown(gauge_html, unsafe_allow_html=True)
            
            # Update Chart (always, to show movement)
            if st.session_state.probability_history:
                chart_data = pd.DataFrame({
                    'Time': st.session_state.time_history,
                    'Probability': st.session_state.probability_history
                })
                # Simple line chart
                if MATPLOTLIB_AVAILABLE:
                     fig, ax = plt.subplots(figsize=(8, 3))
                     ax.plot(st.session_state.probability_history, color='#3498db')
                     ax.set_ylim(0, 1)
                     ax.set_ylabel('PD Probability')
                     ax.set_title('Live Prediction Trend')
                     ax.grid(True, alpha=0.3)
                     chart_placeholder.pyplot(fig)
                     plt.close(fig)
                else:
                    chart_placeholder.line_chart(st.session_state.probability_history)

            # Update Log
            log_html = "#### Recent Predictions\n" + "\n".join([f"- {entry}" for entry in st.session_state.prediction_log])
            log_placeholder.markdown(log_html)
            
            # Force rerun to create a loop effect (standard Streamlit hack for real-time)
            # However, webrtc context handles the loop naturally. We just need to refresh UI.
            # st.experimental_rerun() is deprecated, using run on change or just relying on loop
            time.sleep(0.1) 
            st.rerun()

# === REALTIME MODULE END ===

elif page == "Algorithm Comparison":
    st.header("Algorithm Comparison")
    
    # Overall performance comparison
    st.subheader("Overall Performance")
    
    # Create a DataFrame for comparison
    comparison_data = []
    for algo, results in algorithm_results.items():
        comparison_data.append({
            "Algorithm": algo,
            "Accuracy": results["accuracy"],
            "Features Selected": results["features_selected"],
            "Feature Reduction (%)": round((43 - results["features_selected"]) / 43 * 100, 1),
            "Healthy Precision": results["precision_healthy"],
            "Healthy Recall": results["recall_healthy"],
            "PD Precision": results["precision_pd"],
            "PD Recall": results["recall_pd"]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualization of performance metrics
    st.subheader("Performance Visualization")
    
    if MATPLOTLIB_AVAILABLE:
        try:
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy comparison
            algorithms = list(algorithm_results.keys())
            accuracies = [algorithm_results[algo]["accuracy"] for algo in algorithms]
            ax[0, 0].bar(algorithms, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
            ax[0, 0].set_title("Accuracy Comparison")
            ax[0, 0].set_ylabel("Accuracy")
            ax[0, 0].set_ylim(0, 1)
            
            # Features selected comparison
            features_selected = [algorithm_results[algo]["features_selected"] for algo in algorithms]
            ax[0, 1].bar(algorithms, features_selected, color=['#3498db', '#2ecc71', '#e74c3c'])
            ax[0, 1].set_title("Features Selected")
            ax[0, 1].set_ylabel("Number of Features")
            
            # Precision comparison
            healthy_precision = [algorithm_results[algo]["precision_healthy"] for algo in algorithms]
            pd_precision = [algorithm_results[algo]["precision_pd"] for algo in algorithms]
            x = np.arange(len(algorithms))
            width = 0.35
            ax[1, 0].bar(x - width/2, healthy_precision, width, label='Healthy', color='#3498db')
            ax[1, 0].bar(x + width/2, pd_precision, width, label='Parkinson\'s', color='#e74c3c')
            ax[1, 0].set_title("Precision Comparison")
            ax[1, 0].set_ylabel("Precision")
            ax[1, 0].set_xticks(x)
            ax[1, 0].set_xticklabels(algorithms)
            ax[1, 0].legend()
            
            # Recall comparison
            healthy_recall = [algorithm_results[algo]["recall_healthy"] for algo in algorithms]
            pd_recall = [algorithm_results[algo]["recall_pd"] for algo in algorithms]
            ax[1, 1].bar(x - width/2, healthy_recall, width, label='Healthy', color='#3498db')
            ax[1, 1].bar(x + width/2, pd_recall, width, label='Parkinson\'s', color='#e74c3c')
            ax[1, 1].set_title("Recall Comparison")
            ax[1, 1].set_ylabel("Recall")
            ax[1, 1].set_xticks(x)
            ax[1, 1].set_xticklabels(algorithms)
            ax[1, 1].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)  # Close the figure to free memory
        except Exception as e:
            st.info("Performance visualization requires matplotlib. Install matplotlib to see charts.")
    else:
        st.info("Performance visualization requires matplotlib. Install matplotlib to see charts.")
    
    # Detailed algorithm analysis
    st.subheader("Detailed Algorithm Analysis")
    selected_algorithm = st.selectbox("Select Algorithm for Detailed Analysis", list(algorithm_results.keys()))
    
    if selected_algorithm:
        results = algorithm_results[selected_algorithm]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {selected_algorithm}")
            st.metric("Accuracy", f"{results['accuracy']:.2%}")
            st.metric("Features Selected", results['features_selected'])
            st.metric("Feature Reduction", f"{((43 - results['features_selected']) / 43 * 100):.1f}%")
            
            st.markdown("### Per-Class Performance")
            st.metric("Healthy Precision", f"{results['precision_healthy']:.2%}")
            st.metric("Healthy Recall", f"{results['recall_healthy']:.2%}")
            st.metric("PD Precision", f"{results['precision_pd']:.2%}")
            st.metric("PD Recall", f"{results['recall_pd']:.2%}")
        
        with col2:
            st.markdown("### Confusion Matrix")
            if MATPLOTLIB_AVAILABLE:
                try:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Healthy', 'Parkinson\'s'],
                                yticklabels=['Healthy', 'Parkinson\'s'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'Confusion Matrix - {selected_algorithm}')
                    st.pyplot(fig)
                    plt.close(fig)  # Close the figure to free memory
                except Exception as e:
                    st.write("Confusion Matrix:")
                    st.write(pd.DataFrame(results['confusion_matrix'], 
                                        columns=['Healthy', 'Parkinson\'s'],
                                        index=['Healthy', 'Parkinson\'s']))
            else:
                st.write("Confusion Matrix:")
                st.write(pd.DataFrame(results['confusion_matrix'], 
                                    columns=['Healthy', 'Parkinson\'s'],
                                    index=['Healthy', 'Parkinson\'s']))

elif page == "Feature Analysis":
    st.header("Feature Analysis")
    
    # Feature selection comparison across algorithms
    st.subheader("Feature Selection Comparison")
    
    # Create a matrix showing which features were selected by which algorithms
    feature_matrix = []
    for feature in all_features:
        row = {"Feature": feature}
        for algo in algorithm_results.keys():
            row[algo] = "✓" if feature in algorithm_results[algo]["selected_features"] else "✗"
        feature_matrix.append(row)
    
    feature_df = pd.DataFrame(feature_matrix)
    st.dataframe(feature_df, use_container_width=True, height=800)
    
    # Feature importance visualization
    st.subheader("Feature Importance by Algorithm")
    selected_algo = st.selectbox("Select Algorithm", list(algorithm_results.keys()), key="feature_importance")
    
    if selected_algo:
        importance_data = algorithm_results[selected_algo]["feature_importance"]
        features, importances = zip(*importance_data)
        
        if MATPLOTLIB_AVAILABLE:
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                y_pos = np.arange(len(features))
                ax.barh(y_pos, importances, color='#3498db')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.invert_yaxis()
                ax.set_xlabel('Importance')
                ax.set_title(f'Feature Importance - {selected_algo}')
                st.pyplot(fig)
                plt.close(fig)  # Close the figure to free memory
            except Exception as e:
                st.write(f"Feature Importance - {selected_algo}:")
                importance_df = pd.DataFrame(importance_data, columns=['Feature', 'Importance'])
                st.dataframe(importance_df, use_container_width=True)
        else:
            st.write(f"Feature Importance - {selected_algo}:")
            importance_df = pd.DataFrame(importance_data, columns=['Feature', 'Importance'])
            st.dataframe(importance_df, use_container_width=True)
    
    # Venn diagram of feature selection (simplified)
    st.subheader("Feature Overlap Analysis")
    st.markdown("""
    The following analysis shows how features are shared between different algorithms:
    
    - All algorithms tend to select similar core features like MFCC2_mean and MFCC10_std
    - Each algorithm also selects some unique features
    - The overlap demonstrates consensus on important discriminative features
    """)

elif page == "Model Performance":
    st.header("Model Performance")
    
    st.subheader("Classification Reports")
    
    # Sample classification report data
    report_data = {
        "Gray Wolf Optimization": {
            "Healthy": {"precision": 0.71, "recall": 0.92, "f1-score": 0.80, "support": 13},
            "Parkinson's": {"precision": 0.88, "recall": 0.58, "f1-score": 0.70, "support": 12},
            "accuracy": 0.76,
            "macro avg": {"precision": 0.79, "recall": 0.75, "f1-score": 0.75, "support": 25},
            "weighted avg": {"precision": 0.79, "recall": 0.76, "f1-score": 0.75, "support": 25}
        },
        "Artificial Bee Colony": {
            "Healthy": {"precision": 0.69, "recall": 0.85, "f1-score": 0.76, "support": 13},
            "Parkinson's": {"precision": 0.82, "recall": 0.50, "f1-score": 0.62, "support": 12},
            "accuracy": 0.72,
            "macro avg": {"precision": 0.75, "recall": 0.67, "f1-score": 0.69, "support": 25},
            "weighted avg": {"precision": 0.75, "recall": 0.68, "f1-score": 0.69, "support": 25}
        },
        "Particle Swarm Optimization": {
            "Healthy": {"precision": 0.70, "recall": 0.88, "f1-score": 0.78, "support": 13},
            "Parkinson's": {"precision": 0.85, "recall": 0.53, "f1-score": 0.65, "support": 12},
            "accuracy": 0.74,
            "macro avg": {"precision": 0.77, "recall": 0.71, "f1-score": 0.72, "support": 25},
            "weighted avg": {"precision": 0.77, "recall": 0.72, "f1-score": 0.72, "support": 25}
        }
    }
    
    selected_report = st.selectbox("Select Algorithm for Detailed Report", list(report_data.keys()))
    
    if selected_report:
        report = report_data[selected_report]
        st.markdown(f"### {selected_report} - Classification Report")
        
        # Create a DataFrame for the classification report
        report_df = pd.DataFrame([
            {"Class": "Healthy", "Precision": report["Healthy"]["precision"], 
             "Recall": report["Healthy"]["recall"], "F1-Score": report["Healthy"]["f1-score"], 
             "Support": report["Healthy"]["support"]},
            {"Class": "Parkinson's", "Precision": report["Parkinson's"]["precision"], 
             "Recall": report["Parkinson's"]["recall"], "F1-Score": report["Parkinson's"]["f1-score"], 
             "Support": report["Parkinson's"]["support"]},
            {"Class": "Accuracy", "Precision": "", "Recall": "", 
             "F1-Score": report["accuracy"], "Support": report["Healthy"]["support"] + report["Parkinson's"]["support"]},
            {"Class": "Macro Avg", "Precision": report["macro avg"]["precision"], 
             "Recall": report["macro avg"]["recall"], "F1-Score": report["macro avg"]["f1-score"], 
             "Support": report["macro avg"]["support"]},
            {"Class": "Weighted Avg", "Precision": report["weighted avg"]["precision"], 
             "Recall": report["weighted avg"]["recall"], "F1-Score": report["weighted avg"]["f1-score"], 
             "Support": report["weighted avg"]["support"]}
        ])
        
        st.dataframe(report_df, use_container_width=True)
    
    # ROC Curve analysis (simulated)
    st.subheader("ROC Curve Analysis")
    st.markdown("""
    The Receiver Operating Characteristic (ROC) curve shows the performance of the classification model 
    at various threshold settings. The Area Under the Curve (AUC) provides an aggregate measure of performance.
    
    - **GWO**: AUC = 0.82
    - **ABC**: AUC = 0.78
    - **PSO**: AUC = 0.80
    
    Higher AUC values indicate better model performance.
    """)

elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ### Project Overview
    This application demonstrates the use of machine learning techniques for detecting Parkinson's Disease 
    through speech analysis. The approach leverages audio feature extraction, feature selection using 
    multiple optimization algorithms, and classification with Random Forest.
    
    ### Technologies Used
    - **Python**: Core programming language
    - **Librosa**: Audio feature extraction
    - **Scikit-learn**: Machine learning algorithms
    - **Streamlit**: Web application framework
    - **Matplotlib/Seaborn**: Data visualization
    
    ### Optimization Algorithms
    1. **Gray Wolf Optimization (GWO)**: Mimics the hunting behavior of gray wolves
    2. **Artificial Bee Colony (ABC)**: Simulates the foraging behavior of honey bees
    3. **Particle Swarm Optimization (PSO)**: Based on the social behavior of bird flocking
    
    ### Key Findings
    - All algorithms achieved comparable performance
    - Feature selection significantly improved accuracy (76% vs 64% with all features)
    - Consensus on important features suggests robustness in the approach
    - MFCC features consistently ranked high in importance across algorithms
    
    ### Future Work
    - Integration with real-time audio processing
    - Expansion to larger datasets
    - Implementation of additional optimization algorithms
    - Deployment as a mobile application
    """)
    
    st.markdown("---")
    st.markdown("Developed for educational and research purposes")

# Footer
st.markdown("---")
st.markdown("🏥 Parkinson's Disease Detection System | Machine Learning in Healthcare")