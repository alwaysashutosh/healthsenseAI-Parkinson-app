import React, { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

// Static list of nearby hospitals in Pune
const NEARBY_HOSPITALS = [
  {
    name: "Ruby Hall Clinic",
    address: "40, Sassoon Road, Near Pune Railway Station, Pune",
    phone: "+91 20 6645 5100",
    distance: "3.2 km"
  },
  {
    name: "Jehangir Hospital",
    address: "32, Sassoon Road, Opposite Railway Station, Pune",
    phone: "+91 20 6681 9999",
    distance: "3.5 km"
  },
  {
    name: "Sahyadri Super Speciality Hospital",
    address: "Plot No. 30-C, Karve Road, Deccan Gymkhana, Pune",
    phone: "+91 20 6723 4000",
    distance: "5.1 km"
  }
];

const ResultPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { patientData, prediction } = location.state || {};

  useEffect(() => {
    // Redirect to form if no data
    if (!patientData || !prediction) {
      navigate('/');
    }
  }, [patientData, prediction, navigate]);

  if (!patientData || !prediction) {
    return null;
  }

  // Get probability from prediction (handle different response formats)
  const probability = prediction.probability !== undefined 
    ? prediction.probability 
    : (prediction.Parkinson !== undefined ? prediction.Parkinson * 100 : 0);

  const isHighRisk = probability > 50;

  const handleNewTest = () => {
    navigate('/');
  };

  return (
    <div className="page-container">
      <div className="form-card result-card">
        {/* App Header */}
        <div className="app-header">
          <div className="app-logo">🧠</div>
          <div className="app-title">Parkinson Detection</div>
        </div>

        {/* Step Indicator */}
        <div className="step-indicator">
          <div className="step completed">✓</div>
          <div className="step-line active"></div>
          <div className="step completed">✓</div>
          <div className="step-line active"></div>
          <div className="step active">3</div>
        </div>

        <h1>Analysis Result</h1>

        {/* Patient Info Summary */}
        <div className="patient-summary">
          <h3>Patient Information</h3>
          <p><strong>Name:</strong> {patientData.name}</p>
          <p><strong>Age:</strong> {patientData.age}</p>
          <p><strong>Gender:</strong> {patientData.gender.charAt(0).toUpperCase() + patientData.gender.slice(1)}</p>
        </div>

        {/* Risk Assessment */}
        <div className={`risk-assessment ${isHighRisk ? 'high-risk' : 'low-risk'}`}>
          <h2>Risk Assessment</h2>
          
          <div className="probability-circle">
            <span className="probability-value">{probability.toFixed(1)}%</span>
            <span className="probability-label">Risk Score</span>
          </div>

          {isHighRisk ? (
            <div className="risk-message high-risk-message">
              <h3>High Risk Detected</h3>
              <p>Our analysis indicates a higher probability of Parkinson's disease markers in your voice sample.</p>
              <p className="recommendation">We recommend consulting with a neurologist for further evaluation.</p>
            </div>
          ) : (
            <div className="risk-message low-risk-message">
              <h3>Low Risk</h3>
              <p>Our analysis indicates a low probability of Parkinson's disease markers in your voice sample.</p>
              <p className="recommendation">Continue regular health check-ups and maintain a healthy lifestyle.</p>
            </div>
          )}
        </div>

        {/* Hospital Recommendations (only for high risk) */}
        {isHighRisk && (
          <div className="hospitals-section">
            <h3>Nearby Hospitals</h3>
            <div className="hospitals-list">
              {NEARBY_HOSPITALS.map((hospital, index) => (
                <div key={index} className="hospital-card">
                  <h4>{hospital.name}</h4>
                  <p><strong>Address:</strong> {hospital.address}</p>
                  <p><strong>Phone:</strong> {hospital.phone}</p>
                  <p><strong>Distance:</strong> {hospital.distance}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="disclaimer">
          <p>
            <strong>Disclaimer:</strong> This screening tool is for informational purposes only 
            and does not constitute a medical diagnosis. Please consult with a qualified healthcare 
            provider for proper medical advice and diagnosis.
          </p>
        </div>

        {/* Action Button */}
        <button onClick={handleNewTest} className="btn-primary">
          Start New Test
        </button>
      </div>
    </div>
  );
};

export default ResultPage;
