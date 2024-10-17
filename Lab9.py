import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your CSV file
data = pd.read_csv("train.csv")

# Set up Streamlit dashboard
st.title("Disaster Tweet Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filter Data")
keyword_filter = st.sidebar.selectbox("Select Keyword", options=data['keyword'].dropna().unique(), index=0)
target_filter = st.sidebar.selectbox("Select Target (1: Disaster, 0: Non-Disaster)", options=[0, 1], index=1)

# Filtered Data
filtered_data = data[(data['keyword'] == keyword_filter) & (data['target'] == target_filter)]
st.write("### Filtered Data")
st.write(filtered_data.head())

# Text and Target for modeling
X = data['text']  
y = data['target']

# Preprocess text
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_vectorized = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train models
logreg = LogisticRegression()
dtree = DecisionTreeClassifier()
rf = RandomForestClassifier()

logreg.fit(X_train, y_train)
dtree.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
logreg_pred = logreg.predict(X_test)
dtree_pred = dtree.predict(X_test)
rf_pred = rf.predict(X_test)

# Model Performance
logreg_acc = accuracy_score(y_test, logreg_pred)
dtree_acc = accuracy_score(y_test, dtree_pred)
rf_acc = accuracy_score(y_test, rf_pred)

# Confusion matrices
logreg_cm = confusion_matrix(y_test, logreg_pred)
dtree_cm = confusion_matrix(y_test, dtree_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

# Performance Comparison
st.write("### Model Performance Comparison")
performance_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [logreg_acc, dtree_acc, rf_acc]
})
st.write(performance_df)

# Confusion Matrix for each model
st.write("### Confusion Matrix for Logistic Regression")
fig, ax = plt.subplots()
sns.heatmap(logreg_cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.write("### Confusion Matrix for Decision Tree")
fig, ax = plt.subplots()
sns.heatmap(dtree_cm, annot=True, fmt="d", cmap="Greens", ax=ax)
st.pyplot(fig)

st.write("### Confusion Matrix for Random Forest")
fig, ax = plt.subplots()
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Oranges", ax=ax)
st.pyplot(fig)
