import React from 'react';
import axios from 'axios';
import { useState, useEffect } from 'react';

function DatasetPage() {
  const [patients, setPatients] = useState([]);

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

  return (
    <div className="container">
      <h1 className="title">Patient Dataset</h1>
      <table className="table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Surname</th>
            <th>Age</th>
            <th>Sex</th>
            <th>Weight</th>
            <th>Height</th>
            <th>BMI</th>
            <th>BMI Category</th>
            <th>Surgery Date</th>
            <th>Surgeon Name</th>
            <th>Prescriptions</th>
            <th>Allergies</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {patients.map((patient, index) => (
            <tr key={index}>
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
                <button onClick={() => saveInlineEdit(index)}>Save</button>
                <button 
                  onClick={() => handleDelete(index)}
                  style={{ marginLeft: '5px', backgroundColor: '#dc3545' }}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default DatasetPage;