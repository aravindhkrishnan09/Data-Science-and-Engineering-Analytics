import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Streamlit app title
st.title('Central Limit Theorem Demonstration')

# Set random seed for reproducibility
np.random.seed(42)

# Simulate a population of 10000 EV battery ranges (in km), using exponential distribution
population_size = 10000
battery_range_population = np.random.exponential(scale=280, size=population_size)  # mean ~280 km

# Sidebar for user input
st.sidebar.header('User Input')
sample_size = st.sidebar.slider('Sample Size', min_value=5, max_value=100, value=30, step=5)
num_samples = st.sidebar.slider('Number of Samples', min_value=100, max_value=10000, value=1000, step=100)

# Display population distribution
st.subheader('Population Distribution')
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(battery_range_population, bins=50, kde=True, color='blue', ax=ax)
ax.axvline(x=280, color='red', linestyle='--', label='Mean Range (280 km)')
ax.set_title('Simulated Population of EV Battery Ranges (Exponential Distribution)')
ax.set_xlabel('Battery Range (km)')
ax.set_ylabel('Frequency')
ax.legend()
st.pyplot(fig)

# Function to simulate sampling distribution of sample means
def simulate_sample_means(population, sample_size, num_samples=1000):
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))
    return sample_means

# Simulate sampling distribution
sample_means = simulate_sample_means(battery_range_population, sample_size, num_samples)

# Display sampling distribution
st.subheader('Sampling Distribution of the Mean')
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(sample_means, bins=50, kde=True, color='lightblue', ax=ax)
ax.axvline(x=np.mean(sample_means), color='red', linestyle='--', label='Mean of Sample Means')
ax.axvline(x=np.mean(sample_means) + np.std(sample_means), color='green', linestyle='--', label='Mean + SD')
ax.axvline(x=np.mean(sample_means) - np.std(sample_means), color='green', linestyle='--', label='Mean - SD')
ax.set_title(f'Sampling Distribution of the Mean (Sample Size = {sample_size})')
ax.set_xlabel('Sample Mean of Battery Range (km)')
ax.set_ylabel('Frequency')
ax.legend()

# Add a text box with mean and standard deviation values
textstr = f"Mean: {np.mean(sample_means):.2f}\nSD: {np.std(sample_means):.2f}"
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

st.pyplot(fig)

# Define sample sizes
sample_sizes = [5, 30, 100]

# Streamlit section for sampling distributions for different sample sizes
st.subheader('Sampling Distributions for Different Sample Sizes')

for n in sample_sizes:
    st.write(f"### Sampling Distribution (Sample Size = {n})")
    sample_means = simulate_sample_means(battery_range_population, sample_size=n)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(sample_means, bins=50, kde=True, color='salmon', ax=ax)
    ax.axvline(x=np.mean(sample_means), color='red', linestyle='--', label='Mean of Sample Means')
    ax.axvline(x=np.mean(sample_means) + np.std(sample_means), color='green', linestyle='--', label='Mean + SD')
    ax.axvline(x=np.mean(sample_means) - np.std(sample_means), color='green', linestyle='--', label='Mean - SD')
    ax.legend()
    ax.set_title(f'Sampling Distribution of the Mean (Sample Size = {n})')
    ax.set_xlabel('Sample Mean of Battery Range (km)')
    ax.set_ylabel('Frequency')

    # Add a text box with mean and standard deviation values
    textstr = f"Mean: {np.mean(sample_means):.2f}\nSD: {np.std(sample_means):.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    st.pyplot(fig)