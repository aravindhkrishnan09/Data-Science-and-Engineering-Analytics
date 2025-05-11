import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

def simulate_temp_and_life(seed=0, n=100):
    np.random.seed(seed)
    temp = np.random.normal(35, 5, n)
    battery_life = 1500 - 20 * temp + np.random.normal(0, 30, n)
    return temp, battery_life

def calculate_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

def plot_temp_vs_life(temp, battery_life, correlation):
    fig, ax = plt.subplots()
    sns.scatterplot(x=temp, y=battery_life, ax=ax, label="Observed")
    ax.plot(temp, 1500 - 20 * temp, color='red', label="Regression Line")
    ax.legend()
    ax.set_title(f"Battery Temperature vs Life (Correlation: {correlation:.2f})")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Battery Life (cycles)")
    return fig

def simulate_charging_and_degradation(seed=0, n=100):
    np.random.seed(seed)
    charging_freq = np.random.uniform(1, 3, n)
    deg_rate = 5 + 8 * charging_freq + np.random.normal(0, 2, n)
    return charging_freq, deg_rate

def fit_regression(X, y):
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    pred = model.predict(X.reshape(-1, 1))
    b0 = model.intercept_
    b1 = model.coef_[0]
    r2 = r2_score(y, pred)
    return model, pred, b0, b1, r2

def fit_polynomial_regression(X, y, degree=2):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, y)
    pred = model.predict(X_poly)
    r2 = r2_score(y, pred)
    return model, pred, poly, r2

def plot_charging_vs_degradation(charging_freq, deg_rate, pred_deg):
    fig, ax = plt.subplots()
    ax.scatter(charging_freq, deg_rate, label="Observed")
    ax.plot(charging_freq, pred_deg, color='red', label="Regression Line")
    ax.set_title("Charging Frequency vs Battery Degradation")
    ax.set_xlabel("Charges per Day")
    ax.set_ylabel("Degradation (%)")
    ax.legend()
    return fig

def simulate_DoD_and_cycle_life(seed=0, n=50):
    np.random.seed(seed)
    DoD = np.linspace(20, 100, n)
    cycle_life = 3000 * (1 - DoD / 100)**1.8 + np.random.normal(0, 50, n)
    return DoD, cycle_life

def plot_DoD_vs_cycle_life(DoD, cycle_life):
    fig, ax = plt.subplots()
    ax.plot(DoD, cycle_life, marker='o')
    ax.set_title("Depth of Discharge vs Cycle Life")
    ax.set_xlabel("Depth of Discharge (%)")
    ax.set_ylabel("Cycle Life (Number of Cycles)")
    return fig

# --- Streamlit UI ---
st.title("EV Battery Analysis: Regression & Correlation in Engineering")

tabs = st.tabs(["Main Analysis", "DoD vs Cycle Life Analysis"])

with tabs[0]:

    st.markdown("""
    ### 1. Correlation in EV Battery Analysis
    **Correlation** quantifies the linear relationship between two variables.
    - Example: Battery Temperature vs. Battery Life
    """)

    temp, battery_life = simulate_temp_and_life()
    correlation = calculate_correlation(temp, battery_life)
    fig1 = plot_temp_vs_life(temp, battery_life, correlation)
    st.pyplot(fig1)

    st.markdown("""
    **What does this plot show?**  
    As battery temperature increases, battery life decreases. This strong negative correlation suggests that higher operating temperatures can significantly reduce battery lifespan in EVs.
    """)

    st.markdown("""
    ### 2. Correlation vs Regression
    - Correlation checks if variables move together.
    - Regression predicts one variable from another.
    """)

    st.code("""
    # Correlation
    np.corrcoef(x, y)

    # Regression
    model = LinearRegression().fit(X, y)
    prediction = model.predict(X)
    """, language="python")

    st.markdown("""
    ### 3. Linear Regression: Predicting Battery Degradation
    **Y = β₀ + β₁·X**
    - **Y**: Dependent variable (e.g., Battery degradation)  
    - **β₀**: Intercept (value of Y when X = 0)  
    - **β₁**: Slope (rate of change of Y with respect to X)  
    - **X**: Independent variable (e.g., Charging frequency)  
    """)

    charging_freq, deg_rate = simulate_charging_and_degradation()
    model, pred_deg, b0, b1, r2 = fit_regression(charging_freq, deg_rate)
    fig2 = plot_charging_vs_degradation(charging_freq, deg_rate, pred_deg)
    st.pyplot(fig2)

    st.markdown("""
    **What does this plot show?**  
    There is a clear positive relationship between charging frequency and battery degradation. More frequent charging leads to higher degradation, as shown by the upward trend and the fitted regression line.
    """)

    st.markdown(f"**Regression Equation:** Degradation = {b0:.2f} + {b1:.2f}·Charging_Freq")
    st.markdown(f"**R² Score:** {r2:.2f}")

    st.markdown("""
    ### 4. Importance of Depth of Discharge (DoD)
    - Higher DoD means more capacity used per cycle → more stress → lower cycle life.
    - EV Use Case: Predicting cycle life from DoD.
    """)

    DoD, cycle_life = simulate_DoD_and_cycle_life()
    fig3 = plot_DoD_vs_cycle_life(DoD, cycle_life)
    st.pyplot(fig3)

    st.markdown("""
    **What does this plot show?**  
    As the Depth of Discharge (DoD) increases, the cycle life of the battery drops sharply. Using more of the battery's capacity per cycle reduces the total number of cycles the battery can deliver.
    """)

    st.markdown("""
    ### 5. R² Score in Regression
    - R² tells how well the independent variables explain the variation in the dependent variable.
    - Closer to 1 → better prediction.
    - In EV battery health models, high R² means strong predictive reliability.
    """)

with tabs[1]:
    st.header("DoD vs Cycle Life Analysis")

    st.markdown("""
    #### Data:
    - DoD = [20, 40, 60, 80, 100]
    - cycle_life = [5000, 4000, 3000, 2000, 1500]
    """)

    dod_arr = np.array([20, 40, 60, 80, 100])
    cycle_life_arr = np.array([5000, 4000, 3000, 2000, 1500])

    pearson_corr = calculate_correlation(dod_arr, cycle_life_arr)
    st.markdown(f"**Pearson correlation coefficient:** {pearson_corr:.2f}")

    model_dod, pred_cycle_life, b0_dod, b1_dod, r2_dod = fit_regression(dod_arr, cycle_life_arr)

    pred_50 = model_dod.predict(np.array([[50]]))[0]
    st.markdown(f"**Regression equation:** Cycle Life = {b0_dod:.2f} + {b1_dod:.2f}·DoD")
    st.markdown(f"**Slope:** {b1_dod:.2f}")
    st.markdown(f"**Intercept:** {b0_dod:.2f}")
    st.markdown(f"**Predicted cycle life for DoD=50%:** {pred_50:.2f}")

    # Polynomial regression (degree 2)
    poly_model, poly_pred, poly_feat, poly_r2 = fit_polynomial_regression(dod_arr, cycle_life_arr, degree=2)
    pred_50_poly = poly_model.predict(poly_feat.transform(np.array([[50]])))[0]
    st.markdown(f"**Polynomial Regression (degree 2) R² Score:** {poly_r2:.2f}")
    st.markdown(f"**Predicted cycle life for DoD=50% (Polynomial):** {pred_50_poly:.2f}")

    def plot_dod_regression(dod, cycle_life, pred, poly_pred):
        fig, ax = plt.subplots()
        ax.scatter(dod, cycle_life, color='blue', label='Data Points')
        ax.plot(dod, pred, color='red', label='Linear Regression')
        ax.plot(dod, poly_pred, color='orange', label='Polynomial Regression')
        # DoD=50% predictions
        ax.scatter(50, pred_50, color='green', label='Linear Pred @ DoD=50%')
        ax.scatter(50, pred_50_poly, color='purple', label='Poly Pred @ DoD=50%')
        ax.axhline(y=pred_50, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=pred_50_poly, color='purple', linestyle='--', alpha=0.5)
        ax.set_title('DoD vs Cycle Life')
        ax.set_xlabel('Depth of Discharge (%)')
        ax.set_ylabel('Cycle Life')
        ax.legend(title='Legend', bbox_to_anchor=(0.5, 1.10), loc='lower center', ncol=3)
        return fig

    fig_dod_reg = plot_dod_regression(dod_arr, cycle_life_arr, pred_cycle_life, poly_pred)
    st.pyplot(fig_dod_reg)

    st.markdown("""
    #### Summary:
    #### Interpretation:
    - **Python & Excel** both show a strong negative correlation between DoD and cycle life.
    - The regression line indicates that as DoD increases, cycle life decreases linearly.
    - The predicted cycle life for DoD=50% is close in both methods.
    - The plot visually confirms the negative relationship and the fit of the regression model.
    """)

    # Add table for key points: Prediction vs. Reality, Improve Accuracy, Polynomial Fit
    st.markdown("""
    #### Key Points: Model vs. Reality in Battery Analysis
    """)
    table_data = {
        "Question": ["Prediction vs. Reality", "Improve Accuracy", "Polynomial Fit?"],
        "In Simple Terms": [
            "Model is an educated guess, reality may vary",
            "Use more inputs (features) for better predictions",
            "If the data curves, a curved line fits better"
        ],
        "EV Example": [
            "Model says 2750 cycles, real test = 2900 due to charging conditions",
            "Add temperature, C-rate, battery type, etc.",
            "Cycle life drops faster after 80% DoD → non-linear drop"
        ]
    }
    table_df = pd.DataFrame(table_data)
    st.table(table_df.style.hide(axis="index"))