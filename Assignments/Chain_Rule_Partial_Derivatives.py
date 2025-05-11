import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("EV Battery Degradation – Partial Derivatives & Chain Rule")

# --- Simple Theory Section ---
st.markdown("""
### Partial Derivatives & Chain Rule
Battery degradation D depends on:
- Temperature (T)
- Cycle count (C)

Function: D(T, C) = 0.01·T² + 0.05·C
- ∂D/∂T: Change in D with respect to T
- ∂D/∂C: Change in D with respect to C

If T(t) = 2·t and C(t) = 5·t, then:
dD/dt = ∂D/∂T × dT/dt + ∂D/∂C × dC/dt
""")

# --- Input Section (Simple) ---
t = 3  # Fixed example value for time

# Calculate T and C
temp = 2 * t
cycles = 5 * t

# Partial derivatives
partial_T = 0.02 * temp
partial_C = 0.05

dD_dt = partial_T * 2 + partial_C * 5

# --- Calculation Steps Section ---
st.subheader("Calculation Steps")
st.markdown(f"""
**Partial Derivatives:**
- ∂D/∂T = d/dT [0.01·T² + 0.05·C] = 0.02·T
- ∂D/∂T at T = {temp:.2f} ⇒ 0.02 × {temp:.2f} = {partial_T:.3f}
- ∂D/∂C = d/dC [0.01·T² + 0.05·C] = 0.05

**Chain Rule:**
- dD/dt = ∂D/∂T × dT/dt + ∂D/∂C × dC/dt
- dT/dt = 2, dC/dt = 5
- dD/dt = {partial_T:.3f} × 2 + {partial_C:.2f} × 5 = {dD_dt:.3f} units/hour
""")

# --- Results Section ---
st.header(f"Results at Time t = {t}")
st.markdown(f"""
- Temperature = 2 × {t} = {temp:.2f} °C
- Cycle count = 5 × {t} = {cycles:.2f}

**Partial Derivatives:**
- ∂D/∂T = 0.02 × {temp:.2f} = {partial_T:.3f}
- ∂D/∂C = {partial_C:.2f}

**Total Rate of Degradation (dD/dt):**
dD/dt = {partial_T:.3f} × 2 + {partial_C:.2f} × 5 = {dD_dt:.3f} units/hour
""")

# --- Plot Section ---
time_vals = np.linspace(0, 10, 100)
D_vals = 0.01 * (2 * time_vals)**2 + 0.05 * (5 * time_vals)

fig, ax = plt.subplots()
ax.plot(time_vals, D_vals, label="Degradation over Time", color="blue")
ax.axvline(t, color='gray', linestyle='--', label=f't = {t}')
ax.set_title("Battery Degradation vs Time")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Degradation (units)")
ax.legend()
st.pyplot(fig)

# --- Simple Interpretation ---
st.markdown(f"""
**Interpretation:**
- At t = {t}, degradation rate is {dD_dt:.3f} units/hour.
- If temperature increases faster, degradation rises more quickly.
- Helps estimate wear rate based on known temperature and usage patterns.
""")

# --- Effect of Faster Temperature Increase ---
st.header("Effect of Faster Temperature Increase vs Cycle Count")
st.markdown("""
If temperature increases faster than cycle count (i.e., dT/dt > dC/dt), the total degradation rate **dD/dt** increases more rapidly, since the temperature term has a quadratic effect in the degradation function. This means that controlling temperature rise is crucial for minimizing battery degradation.

Below, we compare two scenarios:
- **Case 1:** dT/dt = 2, dC/dt = 5 (original)
- **Case 2:** dT/dt = 5, dC/dt = 2 (temperature increases faster)
""")

# Calculate dD/dt for both cases over time
time_vals = np.linspace(0, 10, 100)
temp_vals = 2 * time_vals
cycles_vals = 5 * time_vals
partial_T_vals = 0.02 * temp_vals
partial_C = 0.05

dD_dt_case1 = partial_T_vals * 2 + partial_C * 5  # original
dD_dt_case2 = partial_T_vals * 5 + partial_C * 2  # temp increases faster

# Calculate and display Total Rate of Degradation at t for both cases
partial_T_case1 = 0.02 * (2 * t)
dD_dt_case1_t = partial_T_case1 * 2 + partial_C * 5
partial_T_case2 = 0.02 * (2 * t)
dD_dt_case2_t = partial_T_case2 * 5 + partial_C * 2

st.markdown(f"""
**Total Rate of Degradation at t = {t}:**
- Case 1 (dT/dt=2, dC/dt=5): dD/dt = {dD_dt_case1_t:.3f} units/hour
- Case 2 (dT/dt=5, dC/dt=2): dD/dt = {dD_dt_case2_t:.3f} units/hour
""")

fig2, ax2 = plt.subplots()
ax2.plot(time_vals, dD_dt_case1, label="dT/dt=2, dC/dt=5", color="blue")
ax2.plot(time_vals, dD_dt_case2, label="dT/dt=5, dC/dt=2", color="red", linestyle="--")
ax2.set_title("Rate of Degradation (dD/dt) vs Time")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("dD/dt (units/hour)")
ax2.legend()
st.pyplot(fig2)

st.markdown("""
**Observation:**
- When temperature increases faster than cycle count, the degradation rate (red dashed line) is significantly higher, especially as time increases.
- This highlights the importance of temperature management in battery longevity.
""")
