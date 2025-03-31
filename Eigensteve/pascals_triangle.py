import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def generate_pascals_triangle(num_rows):
    """
    Generate Pascal's triangle with the specified number of rows.
    
    Args:
        num_rows: Number of rows to generate (starting from 0)
        
    Returns:
        List of lists representing Pascal's triangle
    """
    triangle = []
    for i in range(num_rows):
        row = [1]  # First element is always 1
        if i > 0:
            prev_row = triangle[i - 1]
            for j in range(len(prev_row) - 1):
                row.append(prev_row[j] + prev_row[j + 1])
            row.append(1)  # Last element is always 1
        triangle.append(row)
    return triangle


def normalize_row(row):
    """
    Normalize a row of Pascal's triangle to sum to 1.
    
    Args:
        row: A row from Pascal's triangle
        
    Returns:
        Normalized row that sums to 1
    """
    return np.array(row) / sum(row)


def plot_comparison(triangle, rows_to_plot=None):
    """
    Plot selected rows of Pascal's triangle against the corresponding
    Gaussian distribution to show the convergence.
    
    Args:
        triangle: Pascal's triangle as a list of lists
        rows_to_plot: List of row indices to plot, or None to use default values
    """
    if rows_to_plot is None:
        # Default rows to plot - choose rows that show the convergence well
        rows_to_plot = [10, 20, 50, 100]
    
    plt.figure(figsize=(12, 10))
    
    for i, n in enumerate(rows_to_plot):
        if n >= len(triangle):
            print(f"Warning: Requested row {n} exceeds triangle size. Skipping.")
            continue
            
        row = triangle[n]
        normalized_row = normalize_row(row)
        
        # X values for plotting the row (centered)
        x = np.arange(len(row)) - (len(row) - 1) / 2
        
        # For the Gaussian: standard deviation for binomial is sqrt(n*p*q)
        # For Pascal's triangle row n, we have binomial with p=q=0.5
        std_dev = np.sqrt(n * 0.5 * 0.5)
        
        # Generate Gaussian PDF with matching mean and variance
        x_continuous = np.linspace(min(x), max(x), 1000)
        gaussian = norm.pdf(x_continuous, 0, std_dev)
        
        # Scale the Gaussian to match the heights
        scaling_factor = max(normalized_row) / max(gaussian)
        gaussian = gaussian * scaling_factor
        
        plt.subplot(len(rows_to_plot), 1, i + 1)
        plt.bar(x, normalized_row, width=0.8, alpha=0.7, label=f"Pascal's Row {n}")
        plt.plot(x_continuous, gaussian, 'r-', linewidth=2, label='Gaussian Approximation')
        plt.title(f"Row {n} of Pascal's Triangle vs. Gaussian Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pascals_triangle_gaussian.png', dpi=300)
    plt.show()


def main():
    # Generate a large Pascal's triangle
    num_rows = 101  # 0 to 100
    triangle = generate_pascals_triangle(num_rows)
    
    # Visualize selected rows
    plot_comparison(triangle)
    
    # Print some info
    print(f"Generated Pascal's triangle with {num_rows} rows (0 to {num_rows-1})")
    print("As the row number increases, the binomial distribution")
    print("approaches a Gaussian (normal) distribution.")


if __name__ == "__main__":
    main()