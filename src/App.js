import React from "react";
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from "react-router-dom";
import "./App.css";
import WelcomePage from "./pages/WelcomePage";
import AddPatientPage from "./pages/AddPatientPage";
import DatasetPage from "./pages/DatasetPage";

function App() {
  const location = useLocation();
  
  return (
    <div className="container">
      <nav>
        <Link to="/" className={`nav-link ${location.pathname === "/" ? "active" : ""}`}>
          Home
        </Link>
        <Link to="/add" className={`nav-link ${location.pathname === "/add" ? "active" : ""}`}>
          Add Patient
        </Link>
        <Link to="/dataset" className={`nav-link ${location.pathname === "/dataset" ? "active" : ""}`}>
          View Dataset
        </Link>
      </nav>

      <Routes>
        <Route path="/" element={<WelcomePage />} />
        <Route path="/add" element={<AddPatientPage />} />
        <Route path="/dataset" element={<DatasetPage />} />
      </Routes>
    </div>
  );
}

// Wrapper component to provide location context
function AppWrapper() {
  return (
    <Router>
      <App />
    </Router>
  );
}

export default AppWrapper;