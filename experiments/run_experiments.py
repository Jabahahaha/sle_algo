import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from generators.matrix_generators import generate_random_matrix, generate_hilbert_matrix, generate_rhs_vector
from algorithms.gaussian_elimination import gaussian_elimination
from algorithms.jacobi_iteration import jacobi_iteration
from algorithms.gauss_seidel import gauss_seidel
from utils.metrics import calculate_metrics

def run_experiments():
    sizes = [3, 10, 50, 100]
    matrix_types = ['random', 'hilbert']
    results = []

    for size in sizes:
        for mtype in matrix_types:
            if mtype == 'random':
                A = generate_random_matrix(size, diagonally_dominant=True)
            elif mtype == 'hilbert':
                A = generate_hilbert_matrix(size)

            b, x_true = generate_rhs_vector(A)

            # Gaussian Elimination
            try:
                x_gauss, time_gauss = gaussian_elimination(A, b)
                residual_gauss, distance_gauss = calculate_metrics(A, x_gauss, b, x_true)
            except np.linalg.LinAlgError:
                residual_gauss, distance_gauss, time_gauss = np.nan, np.nan, np.nan

            # Jacobi Iteration
            x_jacobi, time_jacobi = jacobi_iteration(A, b)
            residual_jacobi, distance_jacobi = calculate_metrics(A, x_jacobi, b, x_true)

            # Gauss-Seidel Method
            x_gs, time_gs = gauss_seidel(A, b)
            residual_gs, distance_gs = calculate_metrics(A, x_gs, b, x_true)

            results.append({
                'size': size,
                'matrix_type': mtype,
                'method': 'Gaussian Elimination',
                'residual': residual_gauss,
                'distance_to_solution': distance_gauss,
                'time': time_gauss
            })
            results.append({
                'size': size,
                'matrix_type': mtype,
                'method': 'Jacobi Iteration',
                'residual': residual_jacobi,
                'distance_to_solution': distance_jacobi,
                'time': time_jacobi
            })
            results.append({
                'size': size,
                'matrix_type': mtype,
                'method': 'Gauss-Seidel',
                'residual': residual_gs,
                'distance_to_solution': distance_gs,
                'time': time_gs
            })

    df_results = pd.DataFrame(results)
    print(df_results)

    # Plotting results with separate plots for random and Hilbert matrices
    sns.set(style="whitegrid")

    # Filter DataFrames for each matrix type
    df_random = df_results[df_results['matrix_type'] == 'random']
    df_hilbert = df_results[df_results['matrix_type'] == 'hilbert']

    # Plot for random matrices
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="size", y="residual", hue="method", markers=True, data=df_random)
    plt.title("Residuals of Solvers by Matrix Size (Random Matrices)")
    plt.xlabel("Matrix Size")
    plt.ylabel("Residual |Ax - b|")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x="size", y="distance_to_solution", hue="method", markers=True, data=df_random)
    plt.title("Distance to Solution by Matrix Size (Random Matrices)")
    plt.xlabel("Matrix Size")
    plt.ylabel("Distance to Real Solution")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x="size", y="time", hue="method", markers=True, data=df_random)
    plt.title("Computation Time by Matrix Size (Random Matrices)")
    plt.xlabel("Matrix Size")
    plt.ylabel("Computation Time (seconds)")
    plt.grid(True)
    plt.show()

    # Plot for Hilbert matrices
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="size", y="residual", hue="method", markers=True, data=df_hilbert)
    plt.title("Residuals of Solvers by Matrix Size (Hilbert Matrices)")
    plt.xlabel("Matrix Size")
    plt.ylabel("Residual |Ax - b|")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x="size", y="distance_to_solution", hue="method", markers=True, data=df_hilbert)
    plt.title("Distance to Solution by Matrix Size (Hilbert Matrices)")
    plt.xlabel("Matrix Size")
    plt.ylabel("Distance to Real Solution")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x="size", y="time", hue="method", markers=True, data=df_hilbert)
    plt.title("Computation Time by Matrix Size (Hilbert Matrices)")
    plt.xlabel("Matrix Size")
    plt.ylabel("Computation Time (seconds)")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_experiments()
