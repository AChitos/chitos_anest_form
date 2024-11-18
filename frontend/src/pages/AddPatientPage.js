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
      <h1 className="title">Προσθήκη Νέου Ασθενή</h1>
      <form className="form" onSubmit={handleSubmit}>
        <label>Όνομα</label>
        <input name="name" placeholder="Όνομα" value={formData.name} onChange={handleFormChange} required />
        
        <label>Επώνυμο</label>
        <input name="surname" placeholder="Επώνυμο" value={formData.surname} onChange={handleFormChange} required />
        
        <label>Ηλικία</label>
        <input name="age" placeholder="Ηλικία" type="number" value={formData.age} onChange={handleFormChange} required />
        
        <label>Φύλο</label>
        <select name="sex" value={formData.sex} onChange={handleFormChange}>
          <option value="Male">Άνδρας</option>
          <option value="Female">Γυναίκα</option>
        </select>
        
        <label>Βάρος (kg)</label>
        <input name="weight" placeholder="Βάρος" type="number" value={formData.weight} onChange={handleFormChange} required />
        
        <label>Ύψος (cm)</label>
        <input name="height" placeholder="Ύψος" type="number" value={formData.height} onChange={handleFormChange} required />
        
        <label>BMI</label>
        <input name="bmi" placeholder="ΔΜΣ" value={formData.bmi} readOnly />
        
        <label>Κατηγορία BMI</label>
        <input name="bmiCategory" placeholder="Κατηγορία ΔΜΣ" value={formData.bmiCategory} readOnly />
        
        <label>Ημερομηνία Χειρουργείου</label>
        <input name="surgery_date" type="date" value={formData.surgery_date} onChange={handleFormChange} required />
        
        <label>Όνομα Χειρουργού</label>
        <input name="surgeon_name" placeholder="Όνομα Χειρουργού" value={formData.surgeon_name} onChange={handleFormChange} required />
        
        <label>Συνταγές</label>
        <textarea name="prescriptions" placeholder="Συ��ταγές" value={formData.prescriptions} onChange={handleFormChange} />
        
        <label>Αλλεργίες</label>
        <textarea name="allergies" placeholder="Αλλεργίες" value={formData.allergies} onChange={handleFormChange} />
        
        <button type="submit">Προσθήκη Ασθενή</button>
      </form>
    </div>
  );
}

export default AddPatientPage;