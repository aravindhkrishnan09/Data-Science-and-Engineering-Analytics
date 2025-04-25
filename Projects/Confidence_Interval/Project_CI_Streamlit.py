import streamlit as st
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Confidence Interval for EV Charging Costs", layout="wide")
st.title("Confidence Statistics for Electric Vehicle (EV) Charging Costs")

# Set random seed for reproducibility
np.random.seed(42)

# Sidebar for user input
st.sidebar.header("User Input")
true_mean = st.sidebar.slider("True Mean ($)", min_value=30, max_value=70, value=50, step=1)
sample_std = st.sidebar.slider("Sample Standard Deviation ($)", min_value=5, max_value=20, value=10, step=1)
sample_size = st.sidebar.slider("Sample Size", min_value=10, max_value=500, value=100, step=10)
confidence_level = st.sidebar.slider("Confidence Level (%)", min_value=90, max_value=99, value=95, step=1)
alpha = 1 - (confidence_level / 100)
z_score = norm.ppf(1 - alpha / 2)


# Expander for each section
with st.expander("Confidence Interval"):
    st.header("Confidence Interval (CI)")
    st.markdown(f"""
    A **Confidence Interval** is the range where we think the real average falls based on a sample.
  
    Surveyed EV owners show an average charging cost of $50.  
    The {confidence_level}% CI is dynamically calculated below based on your inputs.
    """)


    # Calculate Confidence Interval
    margin_of_error = z_score * (sample_std / np.sqrt(sample_size))
    ci_lower = true_mean - margin_of_error
    ci_upper = true_mean + margin_of_error

    st.write(f"{confidence_level}% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # Plot: Confidence Interval on Normal Curve
    st.subheader(f"{confidence_level}% Confidence Interval on a Normal Distribution")
    x = np.linspace(true_mean - 4 * sample_std, true_mean + 4 * sample_std, 500)
    y = (1 / (sample_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - true_mean) / sample_std)**2)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(x, y, label="Normal Distribution", color="blue")
    ax.axvline(ci_lower, color="green", linestyle="--", label=f"CI Lower = ${ci_lower:.2f}")
    ax.axvline(ci_upper, color="green", linestyle="--", label=f"CI Upper = ${ci_upper:.2f}")
    ax.axvline(true_mean, color="red", linestyle=":", label=f"Mean = ${true_mean}")
    ax.fill_between(x, y, where=(x >= ci_lower) & (x <= ci_upper), color="lightgreen", alpha=0.5)
    ax.legend()
    st.pyplot(fig)

with st.expander("Confidence Level"):
    st.header("Confidence Level (CL)")
    st.markdown("""
    The **Confidence Level** tells you how sure you are that the true value is inside your CI.
    A higher CL means a wider CI, which is less precise but more reliable.

    If we did this survey 100 times, about 95 out of 100 times, the confidence interval would include the real average charging cost.
    _Common confidence levels:_
    - 90% → more narrow CI (less confidence)
    - 95% → standard
    - 99% → very wide CI (more confidence)
    """)

    st.markdown(
        """
        <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 10px;">
            <p><strong>A 95% CI means: If we repeated this sampling process 100 times, 95 of those CIs would include the true average.</p>
            <p><strong>Meaning, CI is about long-run coverage over repeated sample.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Simplified Confidence Level Calculation
    num_simulations = 100
    captures = 0
    intervals = []

    for _ in range(num_simulations):
        sample = np.random.normal(true_mean, sample_std, sample_size)
        sample_mean = np.mean(sample)
        me = z_score * (sample_std / np.sqrt(sample_size))
        lower = sample_mean - me
        upper = sample_mean + me
        intervals.append((lower, upper))
        if lower <= true_mean <= upper:
            captures += 1

    # Plotting
    st.subheader(f"Visualizing {num_simulations} Confidence Intervals")
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (low, high) in enumerate(intervals):
        color = "green" if low <= true_mean <= high else "red"
        ax.plot([low, high], [i, i], color=color)
    ax.axvline(true_mean, color="black", linestyle="--", label="True Mean = 50")
    ax.set_title(f"{captures}/{num_simulations} intervals captured the true mean")
    ax.set_xlabel("Value")
    ax.set_ylabel("Sample Index")
    ax.legend()
    st.pyplot(fig)

with st.expander("Margin of Error"):
    st.header("Margin of Error (ME)")
    st.markdown("""
    The **Margin of Error** is the amount your estimate might be off — how much you add or subtract from your sample average to get the confidence interval.

    If our sample average is 50 and the ME is dynamically calculated below, the CI becomes 50 ± ME.
    """)

    st.write(f"Margin of Error: ±${margin_of_error:.2f}")

    # Plot: Effect of Margin of Error
    st.subheader("Effect of Margin of Error on Confidence Interval Width")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axvline(true_mean - margin_of_error, color="orange", linestyle="--", label=f"Lower = ${ci_lower:.2f}")
    ax.axvline(true_mean + margin_of_error, color="orange", linestyle="--", label=f"Upper = ${ci_upper:.2f}")
    ax.axvline(true_mean, color="red", linestyle=":", label=f"Mean = ${true_mean}")
    ax.set_title("Effect of Margin of Error on Confidence Interval")
    ax.fill_between(x, y, where=(x >= ci_lower) & (x <= ci_upper), color="orange", alpha=0.4)
    ax.legend()
    st.pyplot(fig)

with st.expander("Calculation based Tasks"):
    st.header("Calculation based Tasks")
    st.markdown(f"""
                ### What is a t-distribution?
                The t-distribution is used when you:
                - Have a **small sample size** (typically under 30)
                - **Don't know** the population standard deviation  
                It helps you create a confidence interval that's **wider** to reflect more uncertainty.

                ---

                This visual calculates a **{confidence_level}% confidence interval** for the **true average EV range** using the t-distribution.
                """)

    # Default EV data
    default_data = [285, 290, 295, 270, 275, 300, 310, 280, 295, 288]

    ev_ranges = np.array(default_data, dtype=float)
    n = len(ev_ranges)

    mean = np.mean(ev_ranges)
    std_dev = np.std(ev_ranges, ddof=1)
    df = n - 1
    confidence = confidence_level/100
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    margin_error = t_critical * (std_dev / np.sqrt(n))
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error

    st.write(f"Sample Size: {n}")
    st.write(f"Sample Mean: `{mean:.2f} km`")
    st.write(f"Standard Deviation (Sample): `{std_dev:.2f} km`")
    st.write(f"t-Critical Value (df={df}): `{t_critical:.3f}`")
    st.write(f"Margin of Error: `{margin_error:.2f} km`")
    st.success(f"{confidence_level}% Confidence Interval: **({ci_lower:.2f} km, {ci_upper:.2f} km)**")

    # Plot
    st.markdown("### Confidence Interval")
    fig, ax = plt.subplots(figsize=(5, 1))
    ax.errorbar(mean, 0, xerr=margin_error, fmt='o', color='navy', capsize=5)
    ax.axvline(mean, color='green', linestyle='--', label='Mean')
    ax.set_xlim(mean - 3 * margin_error, mean + 3 * margin_error)
    ax.set_yticks([])
    ax.set_xlabel("EV Range (km)")
    ax.set_title(f"{confidence_level}% Confidence Interval")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"### {confidence_level}% Confidence Interval for EV Charging Issues")
    st.markdown("""
    Estimate the **True proportion** of electric vehicles (EVs) likely to face charging station issues.
    """)

    #Input
    total_evs = st.number_input("Total EVs tested:", min_value=1, value=150)
    issue_count = st.number_input("EVs with charging issues:", min_value=0, max_value=total_evs, value=12)

    # Calculation
    if total_evs > 0 and issue_count <= total_evs:
        p_hat = issue_count / total_evs
        z_critical = norm.ppf(0.975)  # 95% confidence
        se = np.sqrt(p_hat * (1 - p_hat) / total_evs)
        margin_error = z_critical * se
        lower = p_hat - margin_error
        upper = p_hat + margin_error

        # Results
        st.write(f"Sample Proportion (𝑝̂): `{p_hat:.4f}`")
        st.write(f"Standard Error: `{se:.4f}`")
        st.write(f"Z-critical value (95%): `{z_critical:.2f}`")
        st.write(f"Margin of Error: `{margin_error:.4f}`")
        st.success(f"{confidence_level}% Confidence Interval: **({lower * 100:.2f}%, {upper * 100:.2f}%)**")

        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.errorbar(p_hat, 0, xerr=margin_error, fmt='o', color='purple', capsize=6)
        ax.axvline(p_hat, color='green', linestyle='--', label='Sample Proportion')
        ax.set_xlim(max(0, p_hat - 3 * margin_error), min(1, p_hat + 3 * margin_error))
        ax.set_yticks([])
        ax.set_xlabel("Proportion")
        ax.set_title(f"{confidence_level}% Confidence Interval for Charging Issues")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Please make sure issue count is less than or equal to total EVs.")

with st.expander("Interpretation & Decision Making"):
    st.header("Interpretation & Decision Making")

    st.markdown(""" **Q1-** An engineer says, “The 95% confidence interval for battery range is (310 km, 325 km), so there’s a 95% chance the true range is in this interval.” Is this statement accurate? If not, rewrite it correctly.""")
    st.markdown(""" **Answer-** _The statement is inaccurate because of the keyword **chance**. Once the data is collected and CI is calculated, we are 95% confident that the mean battery range lies between 310 km and 325 km._""")
    
    st.markdown(""" **Q2-** One-sided vs Two-sided Decision: You are evaluating a new EV model that claims a minimum range of 320 km. Which type of CI would you use—one-sided or two-sided? Why?""")
    st.markdown(""" **Answer-** _A one-sided CI would be appropriate here because we are only interested in whether the range is minimum 320 km (Checking only one direction)_""")


    st.markdown(""" **Q3-** Prediction Interval:
                You calculate the average range of EVs as 310 km with a standard deviation of 12 km. What’s the prediction interval for a new individual EV if n = 25?
            """)
    st.markdown("""
    **Answer-** _The prediction interval is wider than the confidence interval because it accounts for the variability of individual EVs, not just the sample mean._
    - **Confidence Interval (CI)** estimates where the **true mean range** lies based on your sample.
    - **Prediction Interval (PI)** estimates where a **new individual EV's range** could fall — it’s always wider.
    """)
    
    # User inputs
    mean = st.number_input("Sample Mean EV Range (km)", value=310.0)
    std_dev = st.number_input("Sample Standard Deviation (km)", value=12.0)
    n = st.number_input("Sample Size (n)", min_value=2, value=25, step=1)
    confidence = confidence_level/100

    # Degrees of freedom
    df = n - 1

    # t-critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, df)

    # Confidence Interval (CI)
    ci_margin = t_crit * (std_dev / np.sqrt(n))
    ci_lower = mean - ci_margin
    ci_upper = mean + ci_margin

    # Prediction Interval (PI)
    pi_margin = t_crit * std_dev * np.sqrt(1 + 1/n)
    pi_lower = mean - pi_margin
    pi_upper = mean + pi_margin

    # Display values
    st.markdown("### Interval Results")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"{confidence_level}% Confidence Interval (Mean EV Range):\n**({ci_lower:.2f} km, {ci_upper:.2f} km)**")
    with col2:
        st.warning(f"{confidence_level}% Prediction Interval (New EV Range):\n**({pi_lower:.2f} km, {pi_upper:.2f} km)**")

    # Plotting
    fig, ax = plt.subplots(figsize=(7, 2.5))

    # Confidence Interval
    ax.errorbar(mean, 0.25, xerr=ci_margin, fmt='o', color='green', capsize=5, label=f'{confidence_level}% CI (mean)')
    # Prediction Interval
    ax.errorbar(mean, -0.25, xerr=pi_margin, fmt='o', color='orange', capsize=5, label=f'{confidence_level}% PI (new EV)')

    # Labels and aesthetics
    ax.axvline(mean, color='blue', linestyle='--', label='Sample Mean')
    ax.set_xlim(mean - pi_margin*1.5, mean + pi_margin*1.5)
    ax.set_yticks([])
    ax.set_xlabel("EV Range (km)")
    ax.set_title("Confidence Interval vs Prediction Interval")
    ax.legend()
    st.pyplot(fig)

with st.expander("Engineering Application - Compare Two EV Models"):
    st.header(f"Compare Two EV Models using {confidence_level}% Confidence Intervals")

    # Input parameters
    mean_a, std_a, n_a = 295, 10, 20
    mean_b, std_b, n_b = 310, 8, 20
    df = n_a - 1
    confidence = confidence_level/100

    # t critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, df)

    # Calculate CIs
    margin_a = t_crit * (std_a / np.sqrt(n_a))
    ci_a = (mean_a - margin_a, mean_a + margin_a)

    margin_b = t_crit * (std_b / np.sqrt(n_b))
    ci_b = (mean_b - margin_b, mean_b + margin_b)

    # Display CIs
    st.markdown(f"### Confidence Intervals ({confidence_level}%)")
    st.write(f"Model A CI: **({ci_a[0]:.2f}, {ci_a[1]:.2f}) km**")
    st.write(f"Model B CI: **({ci_b[0]:.2f}, {ci_b[1]:.2f}) km**")

    # Conclusion based on overlap
    if ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]:
        st.success("The intervals do NOT overlap — Model B has a significantly higher average range.")
    else:
        st.warning("The intervals overlap — there's no strong evidence of a significant difference.")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.errorbar(mean_a, 1, xerr=margin_a, fmt='o', capsize=5, color='blue', label='Model A')
    ax.errorbar(mean_b, 0, xerr=margin_b, fmt='o', capsize=5, color='green', label='Model B')
    ax.axvline(mean_a, linestyle='--', color='blue', alpha=0.5)
    ax.axvline(mean_b, linestyle='--', color='green', alpha=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Model B", "Model A"])
    ax.set_xlabel("EV Range (km)")
    ax.set_title(f"{confidence_level}% Confidence Intervals for Model A & B")
    ax.legend()
    st.pyplot(fig)

with st.expander("Reflection"):
    st.header("Reflection")
    st.markdown("""How can confidence intervals help improve reliability and performance benchmarks in Electric Vehicle production and deployment?""")
    st.markdown("""
                - **Battery Range** : _Shows the expected range most EVs will get—not just the average—so customers know what to expect._
                - **Charging Time** : _Reveals if charging times vary too much, which might need design improvements._
                - **Failure Rates** : _Estimates how often problems (like charging issues) might happen in real-world use._
                - **Quality Control** : _Checks if each batch of EVs is built consistently and meets quality standards._
                - **Model Comparison** : _Confirms if one EV model truly performs better than another, not just by chance._
                """)

with st.expander("Summary of Key Statistical Metrics"):

    # Define the summary data for the key statistical concepts
    data = [
        {"Metric": "Mean (\u03bc or x\u0304)", "Formula": "x\u0304 = \u03a3x / n", "Description": "Average of all data points in a sample or population.", "EV Example": "Average range of 10 EVs: (285+290+...+288)/10 = 288.8 km"},
        {"Metric": "Standard Deviation (\u03c3 or s)", "Formula": "s = \u221a[\u03a3(x - x\u0304)² / (n-1)]", "Description": "Measures how spread out the values are from the mean.", "EV Example": "Smaller std. dev = more consistent EV range."},
        {"Metric": "Variance (\u03c3\u00b2 or s\u00b2)", "Formula": "s\u00b2 = \u03a3(x - x\u0304)² / (n-1)", "Description": "The square of the standard deviation. Shows overall variability.", "EV Example": "Detects how much EV range varies from the average."},
        {"Metric": "Confidence Interval (CI)", "Formula": "x\u0304 ± z*(\u03c3/\u221an) or x\u0304 ± t*(s/\u221an)", "Description": "Range where the true population mean likely lies.", "EV Example": "95% CI for average EV range: (285 km, 295 km)"},
        {"Metric": "Margin of Error (ME)", "Formula": "z*(\u03c3/\u221an) or t*(s/\u221an)", "Description": "Amount of uncertainty in the estimate of the mean.", "EV Example": "±5 km around the average range estimate."},
        {"Metric": "Confidence Level (CL)", "Formula": "Typically 90%, 95%, or 99%", "Description": "Long-run probability that the CI contains the true mean.", "EV Example": "95% CL means 95 out of 100 samples contain the true range."},
        {"Metric": "t-distribution", "Formula": "t = (x\u0304 - \u03bc) / (s/\u221an)", "Description": "Used for small samples when population std dev is unknown.", "EV Example": "Used to estimate range for new EV models based on few tests."},
        {"Metric": "z-score", "Formula": "z = (x - \u03bc) / \u03c3", "Description": "Standardized value that shows how far a point is from the mean.", "EV Example": "EV with 340 km range when \u03bc=310, \u03c3=10 → z = 3."},
        {"Metric": "p-value", "Formula": "Calculated from test statistic (z or t)", "Description": "Probability of getting the result assuming null hypothesis is true.", "EV Example": "Low p (<0.05) = new EV range is significantly different."},
        {"Metric": "Prediction Interval", "Formula": "x\u0304 ± t*(s\u221a(1 + 1/n))", "Description": "Range where a new individual data point is expected to fall.", "EV Example": "Predict next EV range = 290–330 km, given sample mean and std. dev."}
    ]

    # Convert to DataFrame
    df = pd.DataFrame(data)
    st.header("Summary of Key Statistical Metrics in EV Analysis")

    # Display the DataFrame as a table
    st.dataframe(df, use_container_width=True, hide_index=True)

