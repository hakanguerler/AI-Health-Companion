#!/usr/bin/env python
# coding: utf-8

# In[21]:


import streamlit as st
import pandas as pd
import openai

# Load dataset
def load_data():
    data = pd.read_csv('/Users/hakangurler/Desktop/db_drug_interactions.csv')
    return data

data = load_data()

# Classify severity
def classify_severity(description):
    if 'contraindicated' in description.lower():
        return "Severe"
    elif 'significant' in description.lower() or 'moderate' in description.lower():
        return "Moderate"
    elif 'mild' in description.lower() or 'low risk' in description.lower():
        return "Mild"
    else:
        return "Unknown"

data['Severity'] = data['Interaction Description'].apply(classify_severity)

# Streamlit UI
st.title("Drug Interaction Advisor")
st.sidebar.header("Enter Medications")

# Input medications
medications = st.sidebar.text_input("Enter medications separated by commas (e.g., Aspirin, Ibuprofen):")

if medications:
    meds_list = [med.strip() for med in medications.split(',')]
    st.write(f"Checking interactions for: {', '.join(meds_list)}")
    
    # Check for specific interactions between the input medications
    interactions = data[
        (data['Drug 1'].isin(meds_list) & data['Drug 2'].isin(meds_list)) |
        (data['Drug 2'].isin(meds_list) & data['Drug 1'].isin(meds_list))
    ]

    if not interactions.empty:
        st.write("### Interaction Details")
        for _, row in interactions.iterrows():
            st.write(f"**{row['Drug 1']} and {row['Drug 2']}**")
            st.write(f"Severity: {row['Severity']}")
            st.write(f"Description: {row['Interaction Description']}")
            
            if row['Severity'] in ["Moderate", "Severe"]:
                # Generate explanation using LLM
                prompt = f"Explain in simple terms why taking {row['Drug 1']} and {row['Drug 2']} together might cause problems: {row['Interaction Description']}"
                explanation = "Generated explanation (simulated): Be cautious combining these drugs as it may amplify photosensitivity."
                st.write(f"LLM Explanation: {explanation}")
    else:
        st.write("No interactions found for the entered medications.")
else:
    st.write("Please enter medications to check for interactions.")


# In[ ]:




