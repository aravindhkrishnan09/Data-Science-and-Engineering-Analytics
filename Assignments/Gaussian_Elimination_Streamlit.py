import streamlit as st
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

st.title("Gaussian Elimination Application (3*3 System)")

# Gaussian Elimination: Use Case and Onion Peel Analogy
st.markdown("""
## What is Gaussian Elimination?
Gaussian Elimination is a fundamental algorithm in linear algebra used to solve systems of linear equations. It systematically reduces a system to a simpler form (row echelon form), making it easy to find the solution through back substitution.

### Use Case
Gaussian Elimination is widely used in engineering, physics, computer science, and data analysis whenever you need to solve multiple equations with multiple unknowns. For example, in the context of Electric Vehicles (EVs), it can be used to analyze and optimize power distribution in battery cells, calculate energy consumption across different components, or solve equations related to regenerative braking systems.

### Onion Peel Analogy
Think of Gaussian Elimination like peeling an onion: you remove one layer at a time, simplifying the system step by step. Each row operation is like peeling away a layer, getting closer to the core (the solution). Eventually, you are left with a simple system that can be solved directly.

#### Analogy Breakdown:
1. **Outer Layers**: The initial equations represent the outer layers of the onion. Each equation is a layer that contributes to the overall complexity.

2. **Row Operations**: Just as you peel away the layers of an onion, you apply row operations to eliminate variables and simplify the equations.
            
*Each layer you peel away corresponds to eliminating a variable from the system, just as each row operation in Gaussian Elimination reduces the complexity of the equations.*
""")

def print_matrix_html(matrix, description):
    """Helper function to display matrix in a nice format using streamlit."""
    st.write(f"**{description}:**")
    st.table(matrix)

def gaussian_elimination(A, b):
    """Perform Gaussian Elimination and return intermediate steps"""
    steps = []
    augmented = np.column_stack((A.copy(), b.copy()))
    n = len(A)
    inconsistent = False
    
    # Store initial state
    steps.append({
        'description': "Original Augmented Matrix",
        'matrix': augmented.copy(),
        'details': "This is our starting point - the augmented matrix [A|b] combining our coefficients and constants."
    })
    
    # Forward Elimination Phase
    for pivot_row in range(n):
        # Check if all values in the row are zero
        if np.allclose(augmented[pivot_row, :-1], 0):
            if not np.isclose(augmented[pivot_row, -1], 0):
                steps.append({
                    'description': f"Row {pivot_row} is all zeros except last column (inconsistent)",
                    'matrix': augmented.copy(),
                    'details': f"Row {pivot_row} has all zero coefficients but nonzero constant: inconsistent system."
                })
                inconsistent = True
                break
            else:
                steps.append({
                    'description': f"Row {pivot_row} is all zeros (redundant)",
                    'matrix': augmented.copy(),
                    'details': f"Row {pivot_row} has all zero coefficients and zero constant: redundant equation."
                })
                continue
        # Current pivot element
        pivot = augmented[pivot_row][pivot_row]
        if np.isclose(pivot, 0):
            # Try to swap with a row below that has a nonzero pivot
            swap_row = None
            for r in range(pivot_row + 1, n):
                if not np.isclose(augmented[r][pivot_row], 0):
                    swap_row = r
                    break
            if swap_row is not None:
                augmented[[pivot_row, swap_row]] = augmented[[swap_row, pivot_row]]
                steps.append({
                    'description': f"Swapped Row {pivot_row} with Row {swap_row} due to zero pivot",
                    'matrix': augmented.copy(),
                    'details': f"Pivot in Row {pivot_row} was zero. Swapped with Row {swap_row} to continue elimination."
                })
                pivot = augmented[pivot_row][pivot_row]
            else:
                steps.append({
                    'description': f"Pivot in Row {pivot_row} is zero (cannot swap, singular)",
                    'matrix': augmented.copy(),
                    'details': f"Pivot element is zero and no suitable row to swap. System may be singular or have infinite solutions."
                })
                inconsistent = True
                break
        # Normalize current row
        calculation_str = f"Row {pivot_row} / {pivot:.2f} → Row {pivot_row}"
        augmented[pivot_row] = augmented[pivot_row] / pivot
        steps.append({
            'description': f"After normalizing Row {pivot_row} by pivot ({pivot:.2f})",
            'matrix': augmented.copy(),
            'details': f"Divided Row {pivot_row} by {pivot:.2f} to get a leading 1",
            'calculation': calculation_str
        })
        # Eliminate below
        for row in range(pivot_row + 1, n):
            multiplier = augmented[row][pivot_row]
            calculation_str = f"Row {row} - ({multiplier:.2f}) × Row {pivot_row} → Row {row}"
            augmented[row] = augmented[row] - multiplier * augmented[pivot_row]
            steps.append({
                'description': f"After eliminating Row {row}",
                'matrix': augmented.copy(),
                'details': f"Subtracted {multiplier:.2f} times Row {pivot_row} from Row {row}",
                'calculation': calculation_str
            })
    if inconsistent:
        steps.append({
            'description': "System marked as inconsistent during elimination",
            'matrix': augmented.copy(),
            'details': "Elimination stopped due to inconsistency or zero pivot."
        })
    return steps

# Streamlit UI
st.write("This app demonstrates the Gaussian Elimination method for solving systems of linear equations.")

# Add explanation of basic rules and assumptions
st.markdown("""
### Basic Rules and Assumptions for Gaussian Elimination
- The system should be a set of linear equations with the same number of equations as unknowns (a square system).
- The method uses three types of row operations: swapping rows, multiplying a row by a nonzero constant, and adding/subtracting multiples of one row to another.
- The pivot element (diagonal entry) should not be zero. If it is, row swapping is performed automatically by this app to continue elimination.
- The goal is to convert the system to upper triangular (row echelon) form, then use back substitution to solve for the variables.
- The method assumes the system is consistent and has a unique solution unless a contradiction or redundancy is detected during elimination.
""")

# Input section
st.header("System of Equations")
st.write("Default example:")
st.write("2x + 3y + z = 1")
st.write("4x + y + 2z = 2")
st.write("-2x + 5y - z = 0")

use_custom = st.checkbox("Use custom equations?")

# Add functionality to allow user input of equations in text format
if use_custom:
    st.write("Enter your equations in the format 'ax + by + cz = d' (one per line):")
    equations = st.text_area("Input equations (3 lines for a 3x3 system):", height=100)

    if equations:
        try:
            # Parse equations into A and b
            lines = equations.splitlines()
            A = []
            b = []
            for line in lines:
                # Extract coefficients and constants using regex
                import re
                match = re.findall(r'[-+]?\d*\.?\d+', line)
                if len(match) == 4:  # 3 coefficients + 1 constant
                    A.append([float(match[0]), float(match[1]), float(match[2])])
                    b.append(float(match[3]))
                else:
                    raise ValueError("Each equation must have exactly 3 variables and 1 constant.")

            A = np.array(A)
            b = np.array(b)
        except Exception as e:
            st.error(f"Error parsing equations: {e}")
            st.stop()
else:
    # Default example
    A = np.array([[2, 3, 1], 
                  [4, 1, 2], 
                  [-2, 5, -1]], dtype=float)
    b = np.array([1, 2, 0], dtype=float)

if st.button("Solve"):
    st.header("Solution Steps")
    
    # Get all steps of the solution
    steps = gaussian_elimination(A, b)
    
    # Display each step with explanation
    for idx, step in enumerate(steps):
        with st.expander(f"Step {idx}: {step['description']}", expanded=True):
            st.write(step['details'])
            if 'calculation' in step:
                st.markdown(f'<div style="background-color: #fff3cd; padding: 8px; border-radius: 5px; color: #856404; font-weight: bold;">{step["calculation"]}</div>', unsafe_allow_html=True)
            st.table(step['matrix'])
            
    # Final analysis
    final_matrix = steps[-1]['matrix']
    original_matrix = steps[0]['matrix']

    # Visual comparison
    st.header("Visual Comparison: Input vs Output Matrix")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_matrix, cmap='Blues', aspect='auto')
    axes[0].set_title("Original Augmented Matrix")
    axes[1].imshow(final_matrix, cmap='Greens', aspect='auto')
    axes[1].set_title("Final Matrix (After Elimination)")
    for ax in axes:
        ax.axis('off')
    st.pyplot(fig)

    if np.allclose(final_matrix[-1, :-1], 0) and not np.isclose(final_matrix[-1, -1], 0):
        st.error("The system is INCONSISTENT (no solution exists)")
        st.write("This means there is a contradiction in the equations (0 = non-zero)")
    else:
        st.success("The system has been reduced to upper triangular form")
        st.write("Back substitution can be performed to find the solution")
        
        # Back substitution logic
        def back_substitution(aug_matrix):
            n = aug_matrix.shape[0]
            x = np.zeros(n)
            for i in range(n-1, -1, -1):
                x[i] = aug_matrix[i, -1] - np.dot(aug_matrix[i, i+1:n], x[i+1:n])
            return x
        
        solution = back_substitution(final_matrix)
        st.header("Solution")
        # Default variable names: x, y, z
        variables = ['x', 'y', 'z']
        for idx, val in enumerate(solution):
            st.write(f"{variables[idx]} = {val:.4f}")

    # 3D Visualization of planes (for 3x3 systems)
    if A.shape == (3, 3):
        st.header("Geometric Visualization: Intersection of Planes")
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig3d = plt.figure(figsize=(7, 7))
        ax3d = fig3d.add_subplot(111, projection='3d')
        # Create a grid
        xx, yy = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(3):
            a, b_coef, c = A[i]
            d = b[i]
            if abs(c) > 1e-8:
                zz = (d - a * xx - b_coef * yy) / c
                ax3d.plot_surface(xx, yy, zz, alpha=0.5, color=colors[i], rstride=1, cstride=1, linewidth=0, antialiased=False, label=f'Plane {i+1}')
            else:
                # If c is zero, plot a vertical plane
                continue
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_title('Planes Represented by the System')
        # Try to plot intersection point or line
        if np.allclose(final_matrix[-1, :-1], 0) and not np.isclose(final_matrix[-1, -1], 0):
            st.warning("No intersection: The planes do not all meet (inconsistent system)")
        else:
            # Try to compute intersection point
            try:
                sol = np.linalg.solve(A, b)
                ax3d.scatter([sol[0]], [sol[1]], [sol[2]], color='red', s=80, label='Intersection Point')
            except np.linalg.LinAlgError:
                st.info("Infinite solutions: The planes intersect along a line or coincide.")
        ax3d.legend(title='Planes', bbox_to_anchor=(0.5, -0.20), loc='lower center', ncol=3)
        st.pyplot(fig3d)