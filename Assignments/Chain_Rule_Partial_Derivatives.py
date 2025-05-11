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
