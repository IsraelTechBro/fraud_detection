import os
import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_resources():
    base_folder = os.path.join(os.getcwd(), "resources")
    resource_dict = {}
    for file_name in os.listdir(base_folder):
        full_path = os.path.join(base_folder, file_name)
        name, _ = file_name.split(".")
        resource_dict[name] = joblib.load(full_path)
        
    return resource_dict

resources = load_resources()

st.title("Fraud Detection App")
col1, col2 = st.columns([0.5, 0.5])

with col1:
    # gender
    gender = st.selectbox("select your gender", options=["Male", "Female"])
    gender = {"Male":"M", "Female":"F"}.get(gender)
    # city
    city = st.selectbox("Which city are you in?", options=resources["city"])
    # city_pop
    city_pop = resources["city_pop_map"].get(city)
    # state
    state = st.selectbox("Select your state", options=resources["states"])
    # job
    job = st.selectbox("What's your job?", options=["Others", *resources["jobs"]])
    
with col2:
    # age
    age = st.number_input("What is your age", step=1)
    # amt
    amt = st.number_input("What is the transaction amount?", step=10.0)
    # trans_time
    all_time = [f"{i} AM" for i in range(1, 13)] + [f"{i} PM" for i in range(1, 13)]
    trans_time = st.selectbox("What time of the day did the transaction occur?", options = all_time)
    # category
    category = st.selectbox("Select transaction category", options = ["Others", *resources["categories"]])
    
columns = ["gender", "city", "state", "job", "age", "trans_time", "category", "city_pop", "amt"]
data = [gender, city, state, job, age, trans_time, category, city_pop, amt]

if st.button("Predict Fraud"):
    if all(data):
        X_df = pd.DataFrame(data=[data], columns=columns)
        # encode the df
        pipeline = resources["pipeline"]
        X_enc = pipeline.transform(X_df)

        model = resources["model"] 
        prediction = model.predict(X_enc)[0]
        # handle prediction
        if prediction == 0:
            st.success("Prediction: Non-fraudulent transaction")
        else:
            st.error("Prediction: Fraudulent transaction")
    else:
        st.error("Kindly fill in all the details")
    
