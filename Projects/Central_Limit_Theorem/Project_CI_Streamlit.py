import streamlit as st
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

st.title("Confidence Statistics for Electric Vehicle (EV) Charging Costs")

# Sidebar for user input
st.sidebar.header("User Input")
true_mean = st.sidebar.slider("True Mean ($)", min_value=30, max_value=70, value=50, step=1)
sample_std = st.sidebar.slider("Sample Standard Deviation ($)", min_value=5, max_value=20, value=10, step=1)
sample_size = st.sidebar.slider("Sample Size", min_value=10, max_value=500, value=100, step=10)
confidence_level = st.sidebar.slider("Confidence Level (%)", min_value=90, max_value=99, value=95, step=1)
alpha = 1 - (confidence_level / 100)
z_score = norm.ppf(1 - alpha / 2)


# Tabs for each section
tabs = st.tabs(["Confidence Interval", "Confidence Level", "Margin of Error","Calculation based Tasks","Interpretation & Decision Making"])

# --- a. Confidence Interval ---
with tabs[0]:
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

# --- b. Confidence Level ---
with tabs[1]:
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

# --- c. Margin of Error ---
with tabs[2]:
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

with tabs[3]:
    st.header("Calculation based Tasks")
    st.markdown("""
                ### What is a t-distribution?
                The t-distribution is used when you:
                - Have a **small sample size** (typically under 30)
                - **Don't know** the population standard deviation  
                It helps you create a confidence interval that's **wider** to reflect more uncertainty.

                ---

                This visual calculates a **95% confidence interval** for the **true average EV range** using the t-distribution.
                """)

    # Default EV data
    default_data = [285, 290, 295, 270, 275, 300, 310, 280, 295, 288]

    ev_ranges = np.array(default_data, dtype=float)
    n = len(ev_ranges)

    mean = np.mean(ev_ranges)
    std_dev = np.std(ev_ranges, ddof=1)
    df = n - 1
    confidence = 0.95
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    margin_error = t_critical * (std_dev / np.sqrt(n))
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error

    st.write(f"Sample Size: {n}")
    st.write(f"Sample Mean: `{mean:.2f} km`")
    st.write(f"Standard Deviation (Sample): `{std_dev:.2f} km`")
    st.write(f"t-Critical Value (df={df}): `{t_critical:.3f}`")
    st.write(f"Margin of Error: `{margin_error:.2f} km`")
    st.success(f"95% Confidence Interval: **({ci_lower:.2f} km, {ci_upper:.2f} km)**")

    # Plot
    st.markdown("### Confidence Interval")
    fig, ax = plt.subplots(figsize=(5, 1))
    ax.errorbar(mean, 0, xerr=margin_error, fmt='o', color='navy', capsize=5)
    ax.axvline(mean, color='green', linestyle='--', label='Mean')
    ax.set_xlim(mean - 3 * margin_error, mean + 3 * margin_error)
    ax.set_yticks([])
    ax.set_xlabel("EV Range (km)")
    ax.set_title("95% Confidence Interval")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### 95% Confidence Interval for EV Charging Issues")
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
        st.success(f"95% Confidence Interval: **({lower * 100:.2f}%, {upper * 100:.2f}%)**")

        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.errorbar(p_hat, 0, xerr=margin_error, fmt='o', color='purple', capsize=6)
        ax.axvline(p_hat, color='green', linestyle='--', label='Sample Proportion')
        ax.set_xlim(max(0, p_hat - 3 * margin_error), min(1, p_hat + 3 * margin_error))
        ax.set_yticks([])
        ax.set_xlabel("Proportion")
        ax.set_title("95% Confidence Interval for Charging Issues")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Please make sure issue count is less than or equal to total EVs.")

with tabs[4]:
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
    confidence = 0.95

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
        st.success(f"95% Confidence Interval (Mean EV Range):\n**({ci_lower:.2f} km, {ci_upper:.2f} km)**")
    with col2:
        st.warning(f"95% Prediction Interval (New EV Range):\n**({pi_lower:.2f} km, {pi_upper:.2f} km)**")

    # Plotting
    fig, ax = plt.subplots(figsize=(7, 2.5))

    # Confidence Interval
    ax.errorbar(mean, 0.25, xerr=ci_margin, fmt='o', color='green', capsize=5, label='95% CI (mean)')
    # Prediction Interval
    ax.errorbar(mean, -0.25, xerr=pi_margin, fmt='o', color='orange', capsize=5, label='95% PI (new EV)')

    # Labels and aesthetics
    ax.axvline(mean, color='blue', linestyle='--', label='Sample Mean')
    ax.set_xlim(mean - pi_margin*1.5, mean + pi_margin*1.5)
    ax.set_yticks([])
    ax.set_xlabel("EV Range (km)")
    ax.set_title("Confidence Interval vs Prediction Interval")
    ax.legend()
    st.pyplot(fig)