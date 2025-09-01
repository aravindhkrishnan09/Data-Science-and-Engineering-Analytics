"""
UI Components module for Confidence Interval EV Analysis

This module contains all Streamlit UI components and expandable sections
for the confidence interval analysis application.
"""

import streamlit as st
import pandas as pd
from config import SIMULATION_CONFIG, DEFAULT_EV_DATA, EV_MODEL_DATA
from statistics import (
    calculate_confidence_interval, 
    simulate_confidence_intervals,
    calculate_t_distribution_ci,
    calculate_proportion_confidence_interval,
    calculate_prediction_interval,
    compare_two_models
)
from visualizations import (
    plot_confidence_interval_normal_dist,
    plot_confidence_intervals_simulation,
    plot_margin_of_error_effect,
    plot_t_distribution_ci,
    plot_proportion_ci,
    plot_ci_vs_prediction_interval,
    plot_model_comparison
)

def render_confidence_interval_section(true_mean, sample_std, sample_size, confidence_level):
    """Render the Confidence Interval section"""
    with st.expander("Confidence Interval"):
        st.header("Confidence Interval (CI)")
        st.markdown(f"""
        A **Confidence Interval** is the range where we think the real average falls based on a sample.
      
        Surveyed EV owners show an average charging cost of $50.  
        The {confidence_level}% CI is dynamically calculated below based on your inputs.
        """)

        # Calculate Confidence Interval
        ci_lower, ci_upper, margin_of_error = calculate_confidence_interval(
            true_mean, sample_std, sample_size, confidence_level
        )

        st.write(f"{confidence_level}% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")

        # Plot: Confidence Interval on Normal Curve
        st.subheader(f"{confidence_level}% Confidence Interval on a Normal Distribution")
        plot_confidence_interval_normal_dist(
            true_mean, sample_std, ci_lower, ci_upper, confidence_level
        )

def render_confidence_level_section(true_mean, sample_std, sample_size, confidence_level):
    """Render the Confidence Level section"""
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
        intervals, captures, capture_rate = simulate_confidence_intervals(
            true_mean, sample_std, sample_size, confidence_level, 
            SIMULATION_CONFIG["num_simulations"]
        )

        # Plotting
        st.subheader(f"Visualizing {SIMULATION_CONFIG['num_simulations']} Confidence Intervals")
        plot_confidence_intervals_simulation(
            intervals, true_mean, captures, SIMULATION_CONFIG["num_simulations"]
        )

def render_margin_of_error_section(true_mean, sample_std, sample_size, confidence_level):
    """Render the Margin of Error section"""
    with st.expander("Margin of Error"):
        st.header("Margin of Error (ME)")
        st.markdown("""
        The **Margin of Error** is the amount your estimate might be off — how much you add or subtract from your sample average to get the confidence interval.

        If our sample average is 50 and the ME is dynamically calculated below, the CI becomes 50 ± ME.
        """)

        # Calculate margin of error
        ci_lower, ci_upper, margin_of_error = calculate_confidence_interval(
            true_mean, sample_std, sample_size, confidence_level
        )

        st.write(f"Margin of Error: ±${margin_of_error:.2f}")

        # Plot: Effect of Margin of Error
        st.subheader("Effect of Margin of Error on Confidence Interval Width")
        plot_margin_of_error_effect(
            true_mean, sample_std, ci_lower, ci_upper, margin_of_error
        )

def render_calculation_tasks_section(confidence_level):
    """Render the Calculation based Tasks section"""
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
        ev_ranges = DEFAULT_EV_DATA
        t_dist_results = calculate_t_distribution_ci(ev_ranges, confidence_level)

        st.write(f"Sample Size: {t_dist_results['sample_size']}")
        st.write(f"Sample Mean: `{t_dist_results['mean']:.2f} km`")
        st.write(f"Standard Deviation (Sample): `{t_dist_results['std_dev']:.2f} km`")
        st.write(f"t-Critical Value (df={t_dist_results['degrees_of_freedom']}): `{t_dist_results['t_critical']:.3f}`")
        st.write(f"Margin of Error: `{t_dist_results['margin_error']:.2f} km`")
        st.success(f"{confidence_level}% Confidence Interval: **({t_dist_results['ci_lower']:.2f} km, {t_dist_results['ci_upper']:.2f} km)**")

        # Plot
        st.markdown("### Confidence Interval")
        plot_t_distribution_ci(
            t_dist_results['mean'], 
            t_dist_results['margin_error'], 
            confidence_level
        )

        st.markdown(f"### {confidence_level}% Confidence Interval for EV Charging Issues")
        st.markdown("""
        Estimate the **True proportion** of electric vehicles (EVs) likely to face charging station issues.
        """)

        # Input
        total_evs = st.number_input("Total EVs tested:", min_value=1, value=150)
        issue_count = st.number_input("EVs with charging issues:", min_value=0, max_value=total_evs, value=12)

        # Calculation
        if total_evs > 0 and issue_count <= total_evs:
            lower, upper, margin_error, p_hat, se = calculate_proportion_confidence_interval(
                issue_count, total_evs, confidence_level
            )

            # Results
            st.write(f"Sample Proportion (𝑝̂): `{p_hat:.4f}`")
            st.write(f"Standard Error: `{se:.4f}`")
            st.write(f"Z-critical value (95%): `{1.96:.2f}`")
            st.write(f"Margin of Error: `{margin_error:.4f}`")
            st.success(f"{confidence_level}% Confidence Interval: **({lower * 100:.2f}%, {upper * 100:.2f}%)**")

            plot_proportion_ci(p_hat, margin_error, confidence_level)
        else:
            st.warning("Please make sure issue count is less than or equal to total EVs.")

def render_interpretation_section(confidence_level):
    """Render the Interpretation & Decision Making section"""
    with st.expander("Interpretation & Decision Making"):
        st.header("Interpretation & Decision Making")

        st.markdown(""" **Q1-** An engineer says, "The 95% confidence interval for battery range is (310 km, 325 km), so there's a 95% chance the true range is in this interval." Is this statement accurate? If not, rewrite it correctly.""")
        st.markdown(""" **Answer-** _The statement is inaccurate because of the keyword **chance**. Once the data is collected and CI is calculated, we are 95% confident that the mean battery range lies between 310 km and 325 km._""")
        
        st.markdown(""" **Q2-** One-sided vs Two-sided Decision: You are evaluating a new EV model that claims a minimum range of 320 km. Which type of CI would you use—one-sided or two-sided? Why?""")
        st.markdown(""" **Answer-** _A one-sided CI would be appropriate here because we are only interested in whether the range is minimum 320 km (Checking only one direction)_""")

        st.markdown(""" **Q3-** Prediction Interval:
                    You calculate the average range of EVs as 310 km with a standard deviation of 12 km. What's the prediction interval for a new individual EV if n = 25?
                """)
        st.markdown("""
        **Answer-** _The prediction interval is wider than the confidence interval because it accounts for the variability of individual EVs, not just the sample mean._
        - **Confidence Interval (CI)** estimates where the **true mean range** lies based on your sample.
        - **Prediction Interval (PI)** estimates where a **new individual EV's range** could fall — it's always wider.
        """)
        
        # User inputs
        mean = st.number_input("Sample Mean EV Range (km)", value=310.0)
        std_dev = st.number_input("Sample Standard Deviation (km)", value=12.0)
        n = st.number_input("Sample Size (n)", min_value=2, value=25, step=1)
        confidence = confidence_level/100

        # Calculate intervals
        ci_lower, ci_upper, ci_margin = calculate_confidence_interval(
            mean, std_dev, n, confidence_level, use_t_dist=True
        )
        pi_lower, pi_upper, pi_margin = calculate_prediction_interval(
            mean, std_dev, n, confidence_level
        )

        # Display values
        st.markdown("### Interval Results")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"{confidence_level}% Confidence Interval (Mean EV Range):\n**({ci_lower:.2f} km, {ci_upper:.2f} km)**")
        with col2:
            st.warning(f"{confidence_level}% Prediction Interval (New EV Range):\n**({pi_lower:.2f} km, {pi_upper:.2f} km)**")

        # Plotting
        plot_ci_vs_prediction_interval(mean, ci_margin, pi_margin, confidence_level)

def render_engineering_application_section(confidence_level):
    """Render the Engineering Application section"""
    with st.expander("Engineering Application - Compare Two EV Models"):
        st.header(f"Compare Two EV Models using {confidence_level}% Confidence Intervals")

        # Get model data
        model_a_data = EV_MODEL_DATA["model_a"]
        model_b_data = EV_MODEL_DATA["model_b"]
        
        comparison_results = compare_two_models(
            model_a_data, model_b_data, confidence_level
        )

        # Display CIs
        st.markdown(f"### Confidence Intervals ({confidence_level}%)")
        st.write(f"Model A CI: **({comparison_results['model_a_ci'][0]:.2f}, {comparison_results['model_a_ci'][1]:.2f}) km**")
        st.write(f"Model B CI: **({comparison_results['model_b_ci'][0]:.2f}, {comparison_results['model_b_ci'][1]:.2f}) km**")

        # Conclusion based on overlap
        if comparison_results['no_overlap']:
            st.success("The intervals do NOT overlap — Model B has a significantly higher average range.")
        else:
            st.warning("The intervals overlap — there's no strong evidence of a significant difference.")

        # Plot
        plot_model_comparison(
            comparison_results['model_a_mean'],
            comparison_results['model_b_mean'],
            comparison_results['model_a_ci'][2],  # margin of error
            comparison_results['model_b_ci'][2],  # margin of error
            confidence_level
        )

def render_reflection_section():
    """Render the Reflection section"""
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

def render_summary_section():
    """Render the Summary of Key Statistical Metrics section"""
    with st.expander("Summary of Key Statistical Metrics"):
        # Define the summary data for the key statistical concepts
        data = [
            {"Metric": "Mean (μ or x̄)", "Formula": "x̄ = Σx / n", "Description": "Average of all data points in a sample or population.", "EV Example": "Average range of 10 EVs: (285+290+...+288)/10 = 288.8 km"},
            {"Metric": "Standard Deviation (σ or s)", "Formula": "s = √[Σ(x - x̄)² / (n-1)]", "Description": "Measures how spread out the values are from the mean.", "EV Example": "Smaller std. dev = more consistent EV range."},
            {"Metric": "Variance (σ² or s²)", "Formula": "s² = Σ(x - x̄)² / (n-1)", "Description": "The square of the standard deviation. Shows overall variability.", "EV Example": "Detects how much EV range varies from the average."},
            {"Metric": "Confidence Interval (CI)", "Formula": "x̄ ± z*(σ/√n) or x̄ ± t*(s/√n)", "Description": "Range where the true population mean likely lies.", "EV Example": "95% CI for average EV range: (285 km, 295 km)"},
            {"Metric": "Margin of Error (ME)", "Formula": "z*(σ/√n) or t*(s/√n)", "Description": "Amount of uncertainty in the estimate of the mean.", "EV Example": "±5 km around the average range estimate."},
            {"Metric": "Confidence Level (CL)", "Formula": "Typically 90%, 95%, or 99%", "Description": "Long-run probability that the CI contains the true mean.", "EV Example": "95% CL means 95 out of 100 samples contain the true range."},
            {"Metric": "t-distribution", "Formula": "t = (x̄ - μ) / (s/√n)", "Description": "Used for small samples when population std dev is unknown.", "EV Example": "Used to estimate range for new EV models based on few tests."},
            {"Metric": "z-score", "Formula": "z = (x - μ) / σ", "Description": "Standardized value that shows how far a point is from the mean.", "EV Example": "EV with 340 km range when μ=310, σ=10 → z = 3."},
            {"Metric": "p-value", "Formula": "Calculated from test statistic (z or t)", "Description": "Probability of getting the result assuming null hypothesis is true.", "EV Example": "Low p (<0.05) = new EV range is significantly different."},
            {"Metric": "Prediction Interval", "Formula": "x̄ ± t*(s√(1 + 1/n))", "Description": "Range where a new individual data point is expected to fall.", "EV Example": "Predict next EV range = 290–330 km, given sample mean and std. dev."}
        ]

        # Convert to DataFrame
        df = pd.DataFrame(data)
        st.header("Summary of Key Statistical Metrics in EV Analysis")

        # Display the DataFrame as a table
        st.dataframe(df, use_container_width=True, hide_index=True)
