import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';

import FormPage from './pages/FormPage';
import VoiceInputPage from './pages/VoiceInputPage';
import ResultPage from './pages/ResultPage';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<FormPage />} />
          <Route path="/voice" element={<VoiceInputPage />} />
          <Route path="/result" element={<ResultPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
