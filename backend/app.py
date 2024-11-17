import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CSV_FILE = "patients.csv"


@app.route('/get_patients', methods=['GET'])
def get_patients():
    try:
        if not os.path.exists(CSV_FILE):
            return jsonify([]), 200
        df = pd.read_csv(CSV_FILE)
        return jsonify(df.to_dict(orient="records")), 200
    except Exception as e:
        print("Error retrieving patients:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/add_patient', methods=['POST'])
def add_patient():
    try:
        new_patient = request.json
        df = pd.DataFrame([new_patient])
        if os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(CSV_FILE, index=False)
        return jsonify({"message": "Patient added successfully!"}), 200
    except Exception as e:
        print("Error adding patient:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/update_patient/<int:patient_index>', methods=['PUT'])
def update_patient(patient_index):
    try:
        updated_data = request.json
        if not os.path.exists(CSV_FILE):
            return jsonify({"error": "Dataset not found"}), 404
        df = pd.read_csv(CSV_FILE)
        if patient_index < 0 or patient_index >= len(df):
            return jsonify({"error": "Invalid patient index"}), 400
        for key, value in updated_data.items():
            if key in df.columns:
                df.at[patient_index, key] = value
        df.to_csv(CSV_FILE, index=False)
        return jsonify({"message": "Patient updated successfully!"}), 200
    except Exception as e:
        print("Error updating patient:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/delete_patient/<int:patient_index>', methods=['DELETE'])
def delete_patient(patient_index):
    try:
        if not os.path.exists(CSV_FILE):
            return jsonify({"error": "Dataset not found"}), 404
        df = pd.read_csv(CSV_FILE)
        if patient_index < 0 or patient_index >= len(df):
            return jsonify({"error": "Invalid patient index"}), 400
        df = df.drop(patient_index)
        df.to_csv(CSV_FILE, index=False)
        return jsonify({"message": "Patient deleted successfully!"}), 200
    except Exception as e:
        print("Error deleting patient:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
