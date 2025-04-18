import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title('Central Limit Theorem in EV Systems')


# Set random seed for reproducibility
np.random.seed(42)

#For loop variables
sample_sizes = [5, 30, 100]

# Sidebar for user input
st.sidebar.header('User Input')
population_size = st.sidebar.slider('Population Size', min_value=1000, max_value=100000, value=10000, step=1000)
sample_size = st.sidebar.slider('Sample Size', min_value=5, max_value=100, value=10, step=5)
num_samples = st.sidebar.slider('Number of Samples', min_value=100, max_value=10000, value=10000, step=100)

# Function to simulate sampling distribution of sample means
def simulate_sample_means(population, sample_size, num_samples):
    sample_means = []
    for _ in range(num_samples):
            sample = np.random.choice(population, size=sample_size, replace=True)
            sample_means.append(np.mean(sample))
    return sample_means

# Function to generate sample means
def generate_sample_means(population):
    means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=True)

        if include_drift:
            drift = np.linspace(0, 20, sample_size)
            sample = sample + drift

        if include_autocorrelation:
            for j in range(1, len(sample)):
                sample[j] = 0.7 * sample[j-1] + 0.3 * sample[j]

        means.append(np.mean(sample))
    return means

tabs = st.tabs(["Sampling Battery Range & Effect of Sample Size", "Real-world Scenario Discussion"])
with tabs[0]:
        
        # Simulate a population of 10000 EV battery ranges (in km), using exponential distribution
        battery_range_population = np.random.exponential(scale=280, size=population_size)  # mean ~280 km

        # Display population distribution
        st.subheader("Population Distribution")
        st.markdown("_Population distribution of EV battery ranges, which is generated using an exponential distribution with a mean of approximately 280 km._")
        st.markdown("_The population distribution is skewed to the right, as expected from an exponential distribution with most values concentrated near the lower range and a long tail extending to higher values._")
        st.markdown("_Exponential’s long right tail means there’s a lot of variability, so the standard deviation from the mean is as big as the mean itself._")
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
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

        st.pyplot(fig)

        # Simulate sampling distribution
        sample_means = simulate_sample_means(battery_range_population, sample_size, num_samples)

        # Display sampling distribution
        st.subheader("Sampling Distribution of the Mean")
        st.markdown("_The histogram is approximately normal (bell-shaped), even though the population distribution is skewed. This is a direct result of the Central Limit Theorem, which states that the sampling distribution of the mean approaches normality as the sample size increases._")
        st.markdown("_The mean of the sampling distribution is approximately equal to the mean of the population distribution, and the standard deviation of the sampling distribution (standard error) is smaller than that of the population distribution._")
        st.text(f"Sample Size: {sample_size} \nNumber of Samples: {num_samples}")
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
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

        st.pyplot(fig)

        # Define sample sizes
        sample_sizes = [5, 30, 100]

        # Streamlit section for sampling distributions for different sample sizes
        st.subheader("Sampling Distributions for Different Sample Sizes")
        for n in sample_sizes:
            st.write(f"### Sampling Distribution (Sample Size = {n})")
            sample_means = simulate_sample_means(battery_range_population, sample_size=n, num_samples=num_samples)
            st.text(f"Sample Size: {n} \nNumber of Samples: {num_samples}")
            if n == 5:
                st.markdown("_With a sample size of 5, the sampling distribution is still somewhat skewed, but it is starting to show a more normal shape._")
            elif n == 30:
                st.markdown("_With a sample size of 30, the sampling distribution is more bell-shaped and closely resembles a normal distribution._")
            elif n == 100:
                st.markdown("_With a sample size of 100, the sampling distribution is nearly normal, demonstrating the Central Limit Theorem in action._")
                st.markdown("_As the sample size increases, the standard deviation of the sampling distribution (standard error) decreases, indicating that the sample means are more closely clustered around the population mean allowing us to make accurate predictions based on the sample means._")

            fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
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

            st.pyplot(fig)

with tabs[1]:
     #You want to estimate the average charging time of EVs using only 10 samples. What risks are associated with small samples in engineering decisions?
        st.subheader('Real-world Scenario Discussion')
        st.markdown("_In real-world scenarios, the Central Limit Theorem is crucial for making decisions about population parameters based on sample statistics._")
        st.markdown("_However, using small sample sizes can lead to several risks and challenges in engineering decisions:_")
        st.markdown("- **Increased Variability**: Small samples can lead to high variability in sample means, making it difficult to accurately estimate the population mean.")
        st.markdown("- **Bias**: Small samples may not be representative of the population, leading to biased estimates." )
        st.markdown("- **Inaccurate Confidence Intervals**: The confidence intervals calculated from small samples may be too wide or too narrow, leading to incorrect conclusions.")
        st.markdown("- **Misleading Results**: Small samples can produce misleading results, which can lead to poor engineering decisions and increased costs.")

        st.markdown("_The graph below illustrates the risks associated with small sample sizes in estimating the average charging time of EVs._")

        # plot a seaborn chart to show the average charging time of EVs with only 10 samples
        # Simulate a population of 10000 EV charging times (in hours), using exponential distribution
        charging_time_population = np.random.exponential(scale=8, size=population_size)  # mean ~8 hours
        sample_means = simulate_sample_means(charging_time_population, sample_size, num_samples)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        sns.histplot(sample_means, bins=50, kde=True, color='blue', ax=ax)
        ax.axvline(x=np.mean(sample_means), color='red', linestyle='--', label='Mean of Sample Means: {:.2f}'.format(np.mean(sample_means)))
        ax.axvline(x=np.mean(sample_means) + np.std(sample_means), color='green', linestyle='--', label='Mean + SD: {:.2f}'.format(np.mean(sample_means) + np.std(sample_means)))
        ax.axvline(x=np.mean(sample_means) - np.std(sample_means), color='green', linestyle='--', label='Mean - SD: {:.2f}'.format(np.mean(sample_means) - np.std(sample_means)))
        ax.set_title(f'Sampling Distribution of the Mean (Sample Size = {sample_size})')
        ax.set_xlabel('Sample Mean of Charging Time (hours)')
        ax.set_ylabel('Frequency')
        ax.legend()

         # Add a text box with mean and standard deviation values
        textstr = f"Mean: {np.mean(sample_means):.2f}\nSD: {np.std(sample_means):.2f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        st.pyplot(fig)

        st.subheader("Independent and identically distributed (i.i.d.) vs. Non-i.i.d.")
       
        include_drift = st.checkbox("Include Drift (Non-i.i.d.) - Battery capacity decreasing over time")
        st.markdown("_Drift refers to a gradual change in the mean of the population over time. This can occur due to factors such as sensor aging or environmental changes._")
        include_autocorrelation = st.checkbox("Include Autocorrelation (Non-i.i.d.) - Wheel speed, RPM, and torque are correlated")
        st.markdown("_Autocorrelation refers to the correlation of a variable with itself over successive time intervals. This can occur in time series data where the current value is influenced by previous values.\n" \
        "70% of the previous value and 30% of the original current value_")

        if include_drift or include_autocorrelation:
            st.info("You added **non-i.i.d. behavior**. CLT may **not hold cleanly**. Distributions may be skewed or wider than expected.")
        else:
            st.success("Both populations are **i.i.d.**. The sampling distributions approximate normal curves, showing CLT in action.")

        pop_normal =  np.random.normal(loc=8, scale=10, size=population_size)
        pop_exp = np.random.exponential(scale=8, size=population_size)

        fig_pop, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(pop_normal, bins=50, kde=True, color='skyblue', ax=ax1)
        sns.histplot(pop_exp, bins=50, kde=True, color='orange', ax=ax2)
        ax1.set_title("Normal Distribution")
        ax1.set_xlabel("Charging Time (hours)")
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(pop_normal), color='red', linestyle='--', label='Mean Normal: {:.2f}'.format(np.mean(pop_normal)))
        ax1.axvline(np.mean(pop_normal) + np.std(pop_normal), color='green', linestyle='--', label='Mean + SD: {:.2f}'.format(np.mean(pop_normal) + np.std(pop_normal)))
        ax1.axvline(np.mean(pop_normal) - np.std(pop_normal), color='green', linestyle='--', label='Mean - SD: {:.2f}'.format(np.mean(pop_normal) - np.std(pop_normal)))
        ax2.set_title("Exponential Distribution")
        ax2.set_xlabel("Charging Time (hours)")
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(pop_exp), color='red', linestyle='--', label='Mean Exp: {:.2f}'.format(np.mean(pop_exp)))
        ax2.axvline(np.mean(pop_exp) + np.std(pop_exp), color='green', linestyle='--', label='Mean + SD: {:.2f}'.format(np.mean(pop_exp) + np.std(pop_exp)))
        ax2.axvline(np.mean(pop_exp) - np.std(pop_exp), color='green', linestyle='--', label='Mean - SD: {:.2f}'.format(np.mean(pop_exp) - np.std(pop_exp)))
        ax1.legend()
        ax2.legend()
        st.pyplot(fig_pop)

        # Get sample means
        sample_means_normal = generate_sample_means(pop_normal)
        sample_means_exp = generate_sample_means(pop_exp)

        # Plot sampling distributions side-by-side
        st.subheader("Sampling Distributions of Sample Means")
        fig_samples, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(sample_means_normal, bins=50, kde=True, color='salmon', ax=ax3)
        sns.histplot(sample_means_exp, bins=50, kde=True, color='lightgreen', ax=ax4)

        # Add vertical lines and labels
        ax3.axvline(np.mean(sample_means_normal), color='red', linestyle='--', label='Mean Normal: {:.2f}'.format(np.mean(sample_means_normal)))
        ax3.axvline(np.mean(sample_means_normal) + np.std(sample_means_normal), color='green', linestyle='--', label='Mean + SD: {:.2f}'.format(np.mean(sample_means_normal) + np.std(sample_means_normal)))
        ax3.axvline(np.mean(sample_means_normal) - np.std(sample_means_normal), color='green', linestyle='--', label='Mean - SD: {:.2f}'.format(np.mean(sample_means_normal) - np.std(sample_means_normal)))
        ax4.axvline(np.mean(sample_means_exp), color='red', linestyle='--', label='Mean Exp: {:.2f}'.format(np.mean(sample_means_exp)))
        ax4.axvline(np.mean(sample_means_exp) + np.std(sample_means_exp), color='green', linestyle='--', label='Mean + SD: {:.2f}'.format(np.mean(sample_means_exp) + np.std(sample_means_exp)))
        ax4.axvline(np.mean(sample_means_exp) - np.std(sample_means_exp), color='green', linestyle='--', label='Mean - SD: {:.2f}'.format(np.mean(sample_means_exp) - np.std(sample_means_exp)))
        ax3.set_title("Sampling Dist. from Normal Population")
        ax3.set_xlabel("Sample Mean of Charging Time (hours)")
        ax3.set_ylabel('Frequency')
        ax4.set_title("Sampling Dist. from Exponential Population")
        ax4.set_xlabel("Sample Mean of Charging Time (hours)")
        ax4.set_ylabel('Frequency')
        ax3.legend()
        ax4.legend()
        st.pyplot(fig_samples)

