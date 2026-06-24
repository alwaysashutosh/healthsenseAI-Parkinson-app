import React, { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';

const VoiceInputPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { patientData } = location.state || {};

  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  useEffect(() => {
    // Redirect to form if no patient data
    if (!patientData) {
      navigate('/');
    }
  }, [patientData, navigate]);

  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isRecording]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setRecordedBlob(audioBlob);
        setUploadedFile(null); // Clear uploaded file if any
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError('Unable to access microphone. Please check permissions.');
      console.error('Error accessing microphone:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
      setRecordingTime(0);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file type
      const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/x-wav'];
      if (!validTypes.includes(file.type) && !file.name.endsWith('.wav') && !file.name.endsWith('.mp3')) {
        setError('Please upload a valid audio file (.wav or .mp3)');
        return;
      }
      setUploadedFile(file);
      setRecordedBlob(null); // Clear recorded audio if any
      setError(null);
    }
  };

  const handleSubmit = async () => {
    const audioFile = recordedBlob || uploadedFile;
    if (!audioFile) {
      setError('Please record or upload an audio file first.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    if (recordedBlob) {
      formData.append('file', recordedBlob, 'recording.wav');
    } else {
      formData.append('file', uploadedFile);
    }

    try {
      const response = await axios.post('http://localhost:5000/upload-demo', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      // Navigate to result page with prediction data
      navigate('/result', {
        state: {
          patientData,
          prediction: response.data
        }
      });
    } catch (err) {
      setError('Failed to process audio. Please try again.');
      console.error('Error uploading audio:', err);
    } finally {
      setLoading(false);
    }
  };

  const clearAudio = () => {
    setRecordedBlob(null);
    setUploadedFile(null);
    setRecordingTime(0);
  };

  const hasAudio = recordedBlob || uploadedFile;

  return (
    <div className="page-container">
      <div className="form-card">
        {/* App Header */}
        <div className="app-header">
          <div className="app-logo">🧠</div>
          <div className="app-title">Parkinson Detection</div>
        </div>

        {/* Step Indicator */}
        <div className="step-indicator">
          <div className="step completed">✓</div>
          <div className="step-line active"></div>
          <div className="step active">2</div>
          <div className="step-line"></div>
          <div className="step">3</div>
        </div>

        <h1>Voice Analysis</h1>
        <p className="subtitle">Record or upload your voice sample for AI-powered analysis</p>

        {error && <div className="error-message">{error}</div>}

        {/* Microphone Recording Section */}
        <div className="input-section">
          <h3>Option 1: Record from Microphone</h3>
          <div className="recording-controls">
            {!isRecording ? (
              <button
                onClick={startRecording}
                className="btn-record"
                disabled={loading}
              >
                Start Recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="btn-stop"
                disabled={loading}
              >
                Stop Recording ({formatTime(recordingTime)})
              </button>
            )}
          </div>
          {recordedBlob && (
            <div className="audio-preview">
              <p>Recording saved!</p>
              <audio controls src={URL.createObjectURL(recordedBlob)} />
            </div>
          )}
        </div>

        <div className="divider">OR</div>

        {/* File Upload Section */}
        <div className="input-section">
          <h3>Option 2: Upload Audio File</h3>
          <div className="file-upload">
            <input
              type="file"
              id="audio-upload"
              accept=".wav,.mp3,audio/*"
              onChange={handleFileUpload}
              disabled={loading || isRecording}
            />
            <label htmlFor="audio-upload" className="btn-upload">
              Choose File
            </label>
            {uploadedFile && (
              <span className="file-name">{uploadedFile.name}</span>
            )}
          </div>
        </div>

        {/* Selected Audio Info */}
        {hasAudio && (
          <div className="selected-audio">
            <p>Ready to analyze: {recordedBlob ? 'Recorded audio' : uploadedFile.name}</p>
            <button onClick={clearAudio} className="btn-clear">
              Clear
            </button>
          </div>
        )}

        {/* Submit Button */}
        <button
          onClick={handleSubmit}
          className="btn-primary"
          disabled={!hasAudio || loading}
        >
          {loading ? (
            <>
              <span className="spinner"></span>
              Processing...
            </>
          ) : (
            'Analyze Voice'
          )}
        </button>

        <button
          onClick={() => navigate('/')}
          className="btn-secondary"
          disabled={loading}
        >
          Back to Form
        </button>
      </div>
    </div>
  );
};

export default VoiceInputPage;
