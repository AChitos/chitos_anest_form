import React from 'react';
import { Link } from 'react-router-dom';

function WelcomePage() {
  return (
    <div className="welcome-container">
      <h1 className="title">Welcome to Patient Management System</h1>
      <p className="welcome-text">
        A comprehensive solution for managing patient records and information.
      </p>
      <div className="welcome-actions">
        <Link to="/add" className="welcome-button">Add New Patient</Link>
        <Link to="/dataset" className="welcome-button">View All Patients</Link>
      </div>
    </div>
  );
}

export default WelcomePage;