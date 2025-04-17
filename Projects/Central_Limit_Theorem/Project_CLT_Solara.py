import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import solara

# Set random seed for reproducibility
np.random.seed(42)

# Simulate a population of 10,000 EV battery ranges (in km), using exponential distribution
population_size = 10000
battery_range_population = np.random.exponential(scale=280, size=population_size)  # mean ~280 km

# Function to simulate sampling distribution of sample means
def simulate_sample_means(population, sample_size, num_samples=1000):
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))
    return sample_means

# Function to generate population plot
def get_population_plot():
    plt.figure(figsize=(10, 5))
    sns.histplot(battery_range_population, bins=50, kde=True, color='blue')
    plt.axvline(x=280, color='red', linestyle='--', label='Mean Range (280 km)')
    plt.title('Simulated Population of EV Battery Ranges (Exponential Distribution)')
    plt.xlabel('Battery Range (km)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    return solara.FigureMatplotlib(plt.gcf())

# Function to generate sampling distribution plot
def get_sampling_plot(sample_size):
    sample_means = simulate_sample_means(battery_range_population, sample_size=sample_size)
    plt.figure(figsize=(10, 5))
    sns.histplot(sample_means, bins=50, kde=True, color='lightblue')
    plt.axvline(x=np.mean(sample_means), color='red', linestyle='--', label='Mean of Sample Means')
    plt.axvline(x=np.mean(sample_means) + np.std(sample_means), color='green', linestyle='--', label='Mean + SD')
    plt.axvline(x=np.mean(sample_means) - np.std(sample_means), color='green', linestyle='--', label='Mean - SD')
    plt.legend()
    plt.title(f'Sampling Distribution of the Mean (Sample Size = {sample_size})')
    plt.xlabel('Sample Mean of Battery Range (km)')
    plt.ylabel('Frequency')
    textstr = f"Mean: {np.mean(sample_means):.2f}\nSD: {np.std(sample_means):.2f}"
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.tight_layout()
    return solara.FigureMatplotlib(plt.gcf())

# Reactive state for controlling plot rendering
show_population = solara.reactive(False)
show_sampling = solara.reactive(False)
sample_size = solara.reactive(30)

# Solara component
@solara.component
def Page():
    solara.Markdown("## EV Battery Range Analysis using the Central Limit Theorem")
    
    with solara.Row():
        solara.Button("Show Population Distribution", on_click=lambda: show_population.set(True))
        solara.SliderInt("Sample Size", value=sample_size, min=5, max=100, step=5)
        solara.Button("Show Sampling Distribution", on_click=lambda: show_sampling.set(True))

    if show_population.value:
        solara.Markdown("### Population Distribution")
        get_population_plot()

    if show_sampling.value:
        solara.Markdown("### Sampling Distribution")
        get_sampling_plot(sample_size.value)
