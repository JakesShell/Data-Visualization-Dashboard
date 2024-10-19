import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data creation
np.random.seed(42)
data = {
    'Model': ['Model A', 'Model B', 'Model C', 'Model D'] * 25,
    'Accuracy': np.random.rand(100) * 100,
    'Precision': np.random.rand(100) * 100,
    'Recall': np.random.rand(100) * 100,
    'F1 Score': np.random.rand(100) * 100,
}

df = pd.DataFrame(data)

# Streamlit application
st.title("AI Model Performance Dashboard")
st.write("Explore the performance metrics of different AI models.")

# Sidebar for filtering
st.sidebar.header("Filter Options")
selected_model = st.sidebar.selectbox("Select a Model:", df['Model'].unique())
selected_metric = st.sidebar.selectbox("Select a Metric:", ['Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Filter the DataFrame
filtered_data = df[df['Model'] == selected_model]

# Display metrics
st.subheader(f"Performance Metrics for {selected_model}")
st.write(filtered_data)

# Plotting
st.subheader("Metric Visualization")
fig, ax = plt.subplots()
sns.barplot(x='Model', y=selected_metric, data=filtered_data, palette="Blues", ax=ax)
ax.set_title(f"{selected_metric} of {selected_model}")
st.pyplot(fig)

# Display overall data
st.subheader("Overall Performance Metrics")
st.write(df.groupby('Model').mean().reset_index())

# Overall metrics visualization
st.subheader("Overall Model Performance")
fig2, ax2 = plt.subplots()
sns.barplot(x='Model', y='Accuracy', data=df.groupby('Model').mean().reset_index(), palette="viridis", ax=ax2)
ax2.set_title("Average Accuracy of Models")
st.pyplot(fig2)
