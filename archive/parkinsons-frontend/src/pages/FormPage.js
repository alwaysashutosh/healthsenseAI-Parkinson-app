import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const FormPage = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    age: '',
    gender: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Store form data in state and navigate to voice input page
    navigate('/voice', { state: { patientData: formData } });
  };

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
          <div className="step active">1</div>
          <div className="step-line"></div>
          <div className="step">2</div>
          <div className="step-line"></div>
          <div className="step">3</div>
        </div>

        <h1>Patient Information</h1>
        <h2>Please enter your details to begin the screening</h2>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name">Full Name</label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="Enter your full name"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="age">Age</label>
            <input
              type="number"
              id="age"
              name="age"
              value={formData.age}
              onChange={handleChange}
              placeholder="Enter your age"
              min="1"
              max="120"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="gender">Gender</label>
            <select
              id="gender"
              name="gender"
              value={formData.gender}
              onChange={handleChange}
              required
            >
              <option value="">Select Gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </div>

          <button type="submit" className="btn-primary">
            Next: Voice Input
          </button>
        </form>
      </div>
    </div>
  );
};

export default FormPage;
