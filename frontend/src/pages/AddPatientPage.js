import React, { useState } from "react";
import axios from "axios";

function AddPatientPage() {
  const [formData, setFormData] = useState({
    name: "",
    surname: "",
    age: "",
    sex: "Male",
    weight: "",
    height: "",
    bmi: "",
    bmiCategory: "",
    surgery_date: "",
    surgeon_name: "",
    prescriptions: "",
    allergies: "",
  });

  const calculateBMI = (weight, height) => {
    if (weight && height) {
      const heightInMeters = height / 100;
      const bmi = (weight / (heightInMeters * heightInMeters)).toFixed(1);
      const category = getBMICategory(bmi, formData.age);
      return { bmi, category };
    }
    return { bmi: "", category: "" };
  };

  const getBMICategory = (bmi, age) => {
    if (age < 18) return "Youth BMI - Consult a pediatric BMI chart";
    if (bmi < 18.5) return "Underweight";
    if (bmi < 25) return "Normal weight";
    if (bmi < 30) return "Overweight";
    return "Obese";
  };

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    const updatedData = { ...formData, [name]: value };
    
    if (name === "weight" || name === "height") {
      const { bmi, category } = calculateBMI(
        name === "weight" ? value : formData.weight,
        name === "height" ? value : formData.height
      );
      updatedData.bmi = bmi;
      updatedData.bmiCategory = category;
    }
    
    setFormData(updatedData);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await axios.post("http://127.0.0.1:5000/add_patient", formData);
      alert("Patient added successfully!");
      setFormData({
        name: "",
        surname: "",
        age: "",
        sex: "Male",
        weight: "",
        height: "",
        bmi: "",
        bmiCategory: "",
        surgery_date: "",
        surgeon_name: "",
        prescriptions: "",
        allergies: "",
      });
    } catch (error) {
      console.error("Error adding patient:", error);
      alert("Error adding patient");
    }
  };

  return (
    <div className="container">
      <h1 className="title">Add New Patient</h1>
      <form className="form" onSubmit={handleSubmit}>
        <label>Name</label>
        <input name="name" placeholder="Name" value={formData.name} onChange={handleFormChange} required />
        
        <label>Surname</label>
        <input name="surname" placeholder="Surname" value={formData.surname} onChange={handleFormChange} required />
        
        <label>Age</label>
        <input name="age" placeholder="Age" type="number" value={formData.age} onChange={handleFormChange} required />
        
        <label>Sex</label>
        <select name="sex" value={formData.sex} onChange={handleFormChange}>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>
        
        <label>Weight (kg)</label>
        <input name="weight" placeholder="Weight" type="number" value={formData.weight} onChange={handleFormChange} required />
        
        <label>Height (cm)</label>
        <input name="height" placeholder="Height" type="number" value={formData.height} onChange={handleFormChange} required />
        
        <label>BMI</label>
        <input name="bmi" placeholder="BMI" value={formData.bmi} readOnly />
        
        <label>BMI Category</label>
        <input name="bmiCategory" placeholder="BMI Category" value={formData.bmiCategory} readOnly />
        
        <label>Surgery Date</label>
        <input name="surgery_date" type="date" value={formData.surgery_date} onChange={handleFormChange} required />
        
        <label>Surgeon's Name</label>
        <input name="surgeon_name" placeholder="Surgeon's Name" value={formData.surgeon_name} onChange={handleFormChange} required />
        
        <label>Prescriptions</label>
        <textarea name="prescriptions" placeholder="Prescriptions" value={formData.prescriptions} onChange={handleFormChange} />
        
        <label>Allergies</label>
        <textarea name="allergies" placeholder="Allergies" value={formData.allergies} onChange={handleFormChange} />
        
        <button type="submit">Add Patient</button>
      </form>
    </div>
  );
}

export default AddPatientPage;