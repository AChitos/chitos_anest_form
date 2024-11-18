import React from 'react';
import axios from 'axios';
import { useState, useEffect } from 'react';

function PatientRow({ patient, index, handleInlineChange, saveInlineEdit, handleDelete }) {
  return (
    <tr>
      <td><input value={patient.name || ''} onChange={(e) => handleInlineChange(index, 'name', e.target.value)} /></td>
      <td><input value={patient.surname || ''} onChange={(e) => handleInlineChange(index, 'surname', e.target.value)} /></td>
      <td><input value={patient.age || ''} onChange={(e) => handleInlineChange(index, 'age', e.target.value)} /></td>
      <td><input value={patient.sex || ''} onChange={(e) => handleInlineChange(index, 'sex', e.target.value)} /></td>
      <td><input value={patient.weight || ''} onChange={(e) => handleInlineChange(index, 'weight', e.target.value)} /></td>
      <td><input value={patient.height || ''} onChange={(e) => handleInlineChange(index, 'height', e.target.value)} /></td>
      <td><input value={patient.bmi || ''} onChange={(e) => handleInlineChange(index, 'bmi', e.target.value)} /></td>
      <td><input value={patient.bmiCategory || ''} onChange={(e) => handleInlineChange(index, 'bmiCategory', e.target.value)} /></td>
      <td><input value={patient.surgery_date || ''} onChange={(e) => handleInlineChange(index, 'surgery_date', e.target.value)} /></td>
      <td><input value={patient.surgeon_name || ''} onChange={(e) => handleInlineChange(index, 'surgeon_name', e.target.value)} /></td>
      <td><input value={patient.prescriptions || ''} onChange={(e) => handleInlineChange(index, 'prescriptions', e.target.value)} /></td>
      <td><input value={patient.allergies || ''} onChange={(e) => handleInlineChange(index, 'allergies', e.target.value)} /></td>
      <td>
        <div className="action-buttons">
          <button 
            className="save-button"
            onClick={() => saveInlineEdit(index)}
            title="Αποθήκευση αλλαγών"
          >
            Αποθήκευση
          </button>
          <button 
            className="delete-button"
            onClick={() => handleDelete(index)}
            title="Διαγραφή ασθενή"
          >
            Διαγραφή
          </button>
        </div>
      </td>
    </tr>
  );
}

function DatasetPage() {
  const [patients, setPatients] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');

  const fetchPatients = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/get_patients");
      if (Array.isArray(response.data)) {
        setPatients(response.data);
      } else {
        console.error("Unexpected response format:", response.data);
        setPatients([]);
      }
    } catch (error) {
      console.error("Error fetching patients:", error);
      setPatients([]);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  const handleInlineChange = (index, field, value) => {
    const updatedPatients = [...patients];
    updatedPatients[index][field] = value;
    setPatients(updatedPatients);
  };

  const saveInlineEdit = async (index) => {
    try {
      await axios.put(`http://127.0.0.1:5000/update_patient/${index}`, patients[index]);
      alert("Patient updated successfully!");
    } catch (error) {
      console.error("Error updating patient:", error);
    }
  };

  const handleDelete = async (index) => {
    if (window.confirm("Are you sure you want to delete this patient?")) {
      try {
        await axios.delete(`http://127.0.0.1:5000/delete_patient/${index}`);
        alert("Patient deleted successfully!");
        fetchPatients();
      } catch (error) {
        console.error("Error deleting patient:", error);
        alert("Error deleting patient");
      }
    }
  };

  const filteredPatients = patients.filter(patient =>
    patient.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.surname?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="container">
      <h1 className="title">Βάση Δεδομένων Ασθενών</h1>
      <div className="search-bar">
        <input
          type="text"
          placeholder="Αναζήτηση ασθενή..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>Όνομα</th>
            <th>Επώνυμο</th>
            <th>Ηλικία</th>
            <th>Φύλο</th>
            <th>Βάρος</th>
            <th>Ύψος</th>
            <th>BMI</th>
            <th>Κατηγορία BMI</th>
            <th>Ημερομηνία Χειρουργείου</th>
            <th>Όνομα Χειρουργού</th>
            <th>Συνταγές</th>
            <th>Αλλεργίες</th>
            <th>Ενέργειες</th>
          </tr>
        </thead>
        <tbody>
          {filteredPatients.map((patient, index) => (
            <PatientRow 
              key={index} 
              patient={patient} 
              index={index} 
              handleInlineChange={handleInlineChange} 
              saveInlineEdit={saveInlineEdit} 
              handleDelete={handleDelete} 
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default DatasetPage;