import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load and train the model (runs once)
@st.cache_resource
def train_model():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf, iris

model, iris = train_model()

# UI
st.title("Iris Flower Species Predictor")
st.write("Input flower measurements to classify the species.")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Predict
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(features)
species = iris.target_names

st.subheader("Prediction")
st.success(f"The predicted species is **{species[prediction[0]]}**.")

# Visualization
st.subheader("Feature Importance")
feature_names = iris.feature_names
importances = model.feature_importances_

fig, ax = plt.subplots()
sns.barplot(x=importances, y=feature_names, ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)
