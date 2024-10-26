import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('gb_model.pkl', 'rb'))

# Define mappings for label-encoded values
vehicle_mapping = {0: "Car", 1: "Truck", 2: "Motorcycle", 3: "Bicycle", 4: "Pedestrian"}
weather_mapping = {0: "Clear", 1: "Rain", 2: "Fog", 3: "Snow"}
cause_mapping = {0: "Over Speeding", 1: "Drunk Driving", 2: "Distracted Driving", 3: "Mechanical Failure", 4: "Road Conditions"}
severity_mapping = {0: "Low", 1: "Moderate", 2: "Severe"}

# Set the title and description
st.title("Injury Severity Prediction website")
st.write("Predict the severity of injuries from road accidents based on various input features.")

# Collect input data from user
st.subheader("Enter Details for Prediction")

vehicle_involved = st.selectbox("Vehicle Involved", list(vehicle_mapping.values()))
weather_conditions = st.selectbox("Weather Conditions", list(weather_mapping.values()))
cause = st.selectbox("Cause", list(cause_mapping.values()))

num_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=10, step=1)
num_injuries = st.number_input("Number of Injuries", min_value=0, max_value=10, step=1)
num_fatalities = st.number_input("Number of Fatalities", min_value=0, max_value=10, step=1)

# Convert selected values to their encoded forms for prediction
vehicle_involved_encoded = list(vehicle_mapping.keys())[list(vehicle_mapping.values()).index(vehicle_involved)]
weather_conditions_encoded = list(weather_mapping.keys())[list(weather_mapping.values()).index(weather_conditions)]
cause_encoded = list(cause_mapping.keys())[list(cause_mapping.values()).index(cause)]

# Predict button
if st.button("Predict Injury Severity"):
    # Create feature array for prediction
    features = np.array([[vehicle_involved_encoded, weather_conditions_encoded, cause_encoded, num_vehicles, num_injuries, num_fatalities]])
    
    # Get prediction
    prediction = model.predict(features)
    severity_level = severity_mapping[int(prediction[0])]

    # Display the results
    st.write("### Prediction Results")
    st.write(f"The predicted Injury Severity is: **{severity_level}**")

    # Display the input details
    st.write("#### Input Details")
    st.write(f"- Vehicle Involved: {vehicle_involved}")
    st.write(f"- Weather Conditions: {weather_conditions}")
    st.write(f"- Cause of Accident: {cause}")
    st.write(f"- Number of Vehicles: {num_vehicles}")
    st.write(f"- Number of Injuries: {num_injuries}")
    st.write(f"- Number of Fatalities: {num_fatalities}")
