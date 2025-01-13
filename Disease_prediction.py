#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,confusion_matrix,precision_score,roc_curve
import seaborn as sns
from sklearn.utils import shuffle
# from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[118]:


#df = pd.read_csv('dataset.csv')
df = pd.read_csv('/Users/hakangurler/Desktop/final_DSR /diease_prediction/archive/dataset.csv')

df.head()


# In[119]:


df.isnull().sum()


# In[120]:


df = df.fillna("")
df['Symptom']=""
for i in range(1,18):
    df['s']=df["Symptom_{}".format(i)]
    df['Symptom']=df['Symptom']+df['s']


# In[121]:


df.head()


# In[122]:


for i in range(1,18):
    df=df.drop("Symptom_{}".format(i),axis=1)
df=df.drop("s",axis=1)


# In[123]:


df.head()


# In[124]:


df['Disease'].value_counts()


# In[125]:


X=df['Symptom']
y=df['Disease']


# In[126]:


X.head(10)


# In[127]:


y.head(10)


# In[128]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,shuffle=True,random_state=44)


# In[129]:


y_train.head(10)


# In[130]:


print('Training Data Shape:', X_train.shape)
print('Testing Data Shape: ', X_test.shape)


# In[131]:


y_train.value_counts()


# In[132]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_train_tfidf.shape


# In[133]:


pd.DataFrame(X_train_tfidf.toarray())


# In[134]:


from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train_tfidf,y_train)


# In[135]:


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),])


text_clf.fit(X_train, y_train)  


predictions = text_clf.predict(X_test)


# In[136]:


from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))


# In[137]:


print(metrics.classification_report(y_test,predictions)) 


# In[138]:


print(metrics.accuracy_score(y_test,predictions))


# In[139]:


#df1 = pd.read_csv('symptom_Description.csv')
df1 = pd.read_csv('/Users/hakangurler/Desktop/final_DSR /diease_prediction/archive/symptom_Description.csv')
df1.head()


# In[140]:


df1.isnull().sum()


# In[141]:


df1.index=df1['Disease']
df1=df1.drop('Disease',axis=1)
df1.head(10)


# In[142]:


#df2 = pd.read_csv('symptom_precaution.csv')
df2 = pd.read_csv('/Users/hakangurler/Desktop/final_DSR /diease_prediction/archive/symptom_precaution.csv')
df2.head()


# In[143]:


df2.isnull().sum()


# In[144]:


df2 = df2.fillna("")
df2.head(10)


# In[145]:


df2['precautions']=""
df2['punc']=', '
for i in range(1,5):
    df2['s']=df2["Precaution_{}".format(i)]+df2['punc']
    df2['precautions']=df2['precautions']+df2['s']
df2.head(10)


# In[146]:


for i in range(1,5):
    df2=df2.drop("Precaution_{}".format(i),axis=1)
df2.head(10) 


# In[147]:


df2=df2.drop(['s','punc'],axis=1)
df2.head(10)


# In[148]:


df2.index=df2['Disease']
df2=df2.drop('Disease',axis=1)
df2.head()


# In[151]:


import streamlit as st
import pandas as pd

# Load the processed data from your dataset
def load_data():
    # Replace with the path to your dataset file
    data = pd.read_csv("/Users/hakangurler/Desktop/Final_Disease_Dset.csv")
    return data

# Title
st.title("Symptom-Based Disease Finder")

# Load data
df = load_data()

# Ensure that all required columns are available
required_columns = {"Disease", "Symptoms", "Precautions"}
if required_columns.issubset(df.columns):
    # User input for symptoms
    st.write("### Input Symptoms")
    user_symptoms = st.text_input("Enter symptoms separated by commas (e.g., fever, headache):")
    
    if user_symptoms:
        # Process user input
        input_symptoms = set([symptom.strip().lower() for symptom in user_symptoms.split(",")])
        
        # Find diseases that match the symptoms
        matched_diseases = []
        for _, row in df.iterrows():
            disease_symptoms = set([symptom.strip().lower() for symptom in row["Symptoms"].split(",")])
            if input_symptoms.intersection(disease_symptoms):  # Check for symptom overlap
                matched_diseases.append(row)

        # Display results
        if matched_diseases:
            st.write("### Possible Diseases Based on Your Symptoms:")
            for disease in matched_diseases:
                st.write(f"- **{disease['Disease']}**")
                st.write(f"  - Symptoms: {disease['Symptoms']}")
                st.write(f"  - Precautions: {disease['Precautions']}")
                st.write("---")
        else:
            st.write("No diseases match the entered symptoms. Please try different symptoms.")
else:
    st.write(f"Error: The dataset must contain the following columns: {', '.join(required_columns)}")


# In[152]:


import streamlit as st
import pandas as pd

# Title
st.title("Disease Precaution Finder")

# Description
st.write("""
This application allows users to search for diseases and view the recommended precautions.
It is built to help raise awareness about various diseases and promote health and safety.
""")

# Sidebar Upload
st.sidebar.header("Upload Processed Data")
uploaded_file = st.sidebar.file_uploader("Upload your processed disease-precaution CSV file", type=["csv"])

if uploaded_file:
    # Load the data
    df = pd.read_csv(uploaded_file, index_col=0)

    # Sidebar for selecting a disease
    disease = st.sidebar.selectbox("Select a Disease", df.index)

    # Display Precautions
    if disease:
        st.subheader(f"Precautions for {disease}")
        precautions = df.loc[disease, "precautions"]
        for i, precaution in enumerate(precautions.split(','), 1):
            st.write(f"{i}. {precaution.strip()}")

    # Display all diseases in the main page
    st.subheader("All Diseases and Precautions")
    st.dataframe(df)

else:
    st.info("Please upload a processed data file to proceed.")
    st.write("Hint: The file should be a CSV where one column is 'Disease' and another is 'precautions'.")

# Footer
st.write("---")
st.write("Made with ❤️ using Streamlit")


# In[ ]:




