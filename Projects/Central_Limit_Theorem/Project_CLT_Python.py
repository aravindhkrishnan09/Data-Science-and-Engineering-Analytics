import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Simulate a population of 10000 EV battery ranges (in km), using exponential distribution
population_size = 10000
battery_range_population = np.random.exponential(scale=280, size=population_size)  # mean ~280 km

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(battery_range_population, bins=50, kde=True, color='blue', ax=ax)
ax.axvline(x= 280, color='red', linestyle='--', label=f'Mean Range: 280 km')
ax.axvline(x= 280 + np.std(battery_range_population), color='green', linestyle='--', label=f'Mean + SD: {280 + np.std(battery_range_population):.2f}')
ax.axvline(x= 280 - np.std(battery_range_population), color='green', linestyle='--', label=f'Mean - SD: {280 - np.std(battery_range_population):.2f}')
ax.set_title('Simulated Population of EV Battery Ranges (Exponential Distribution)')
ax.set_xlabel('Battery Range (km)')
ax.set_ylabel('Frequency')
ax.legend()

# Add a text box with mean and standard deviation values
textstr = f"Mean: 280 \nSD: {np.std(battery_range_population):.2f}"
ax.text(0.80, 0.75, textstr, transform=ax.transAxes, fontsize=12,
        horizontalalignment='left', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),)

plt.show()

# Function to simulate sampling distribution of sample means
def simulate_sample_means(population, sample_size, num_samples):
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))
    return sample_means

# Simulate sampling distribution
sample_size = 30
num_samples = 10000
sample_means = simulate_sample_means(battery_range_population, sample_size, num_samples)

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(sample_means, bins=50, kde=True, color='salmon', ax=ax)
ax.axvline(x=np.mean(sample_means), color='red', linestyle='--', label='Mean of Sample Means: {:.2f}'.format(np.mean(sample_means)))
ax.axvline(x=np.mean(sample_means) + np.std(sample_means), color='green', linestyle='--', label='Mean + SD: {:.2f}'.format(np.mean(sample_means) + np.std(sample_means)))
ax.axvline(x=np.mean(sample_means) - np.std(sample_means), color='green', linestyle='--', label='Mean - SD: {:.2f}'.format(np.mean(sample_means) - np.std(sample_means)))
ax.set_title(f'Sampling Distribution of the Mean (Sample Size = {sample_size})')
ax.set_xlabel('Sample Mean of Battery Range (km)')
ax.set_ylabel('Frequency')
ax.legend()

# Add a text box with mean and standard deviation values
textstr = f"Mean: {np.mean(sample_means):.2f}\nSD: {np.std(sample_means):.2f}"
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.show()

# Define sample sizes
sample_sizes = [5, 30, 100]

num_samples = 10000
for n in sample_sizes:
    sample_means = simulate_sample_means(battery_range_population, sample_size=n, num_samples=num_samples)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(sample_means, bins=50, kde=True, color='salmon', ax=ax)
    ax.axvline(x=np.mean(sample_means), color='red', linestyle='--', label='Mean of Sample Means: {:.2f}'.format(np.mean(sample_means)))
    ax.axvline(x=np.mean(sample_means) + np.std(sample_means), color='green', linestyle='--', label='Mean + SD: {:.2f}'.format(np.mean(sample_means) + np.std(sample_means)))
    ax.axvline(x=np.mean(sample_means) - np.std(sample_means), color='green', linestyle='--', label='Mean - SD: {:.2f}'.format(np.mean(sample_means) - np.std(sample_means)))
    ax.legend()
    ax.set_title(f'Sampling Distribution of the Mean (Sample Size = {n})')
    ax.set_xlabel('Sample Mean of Battery Range (km)')
    ax.set_ylabel('Frequency')

    # Add a text box with mean and standard deviation values
    textstr = f"Mean: {np.mean(sample_means):.2f}\nSD: {np.std(sample_means):.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.show()