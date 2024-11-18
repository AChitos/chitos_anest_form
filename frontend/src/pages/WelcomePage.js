import React from 'react';
import { Link } from 'react-router-dom';

function WelcomePage() {
  return (
    <div className="welcome-container">
      <h1 className="title">Καλώς ήρθατε στο Σύστημα Διαχείρισης Ασθενών</h1>
      <p className="welcome-text">
        Μια ολοκληρωμένη λύση για τη διαχείριση αρχείων και πληροφοριών ασθενών.
      </p>
      <div className="welcome-actions">
        <Link to="/add" className="welcome-button">Προσθήκη Νέου Ασθενή</Link>
        <Link to="/dataset" className="welcome-button">Προβολή Όλων των Ασθενών</Link>
      </div>
    </div>
  );
}

export default WelcomePage;