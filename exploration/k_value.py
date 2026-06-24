import pandas as pd
import matplotlib.pyplot as plt
import os

# Consolidated and sorted data from your tests
k_values = [
    300, 400, 500, 600, 700, 725, 750, 775, 800, 810, 818, 825, 
    850, 875, 900, 1000, 1100, 1200, 1300, 1400, 1500
]

rmse_values = [
    14.9095, 14.4705, 14.3667, 14.2621, 14.1354, 14.0536, 14.0403, 14.0285, 14.0697, 14.0849, 14.0838, 14.0719, 
    14.1479, 14.0992, 14.1007, 14.1976, 14.3025, 14.5161, 14.6027, 14.6396, 15.0850
]

# Set up the plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, rmse_values, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6, label='Validation RMSE')

# Highlight the absolute minimum (Best K)
best_k = 775
best_rmse = 14.0285
plt.scatter([best_k], [best_rmse], color='red', s=120, zorder=5, edgecolor='black', label=f'Optimal K = {best_k}\n(RMSE: {best_rmse:.4f})')

# Add an arrow pointing to the optimal value
plt.annotate('Lowest Error', 
             xy=(best_k, best_rmse), 
             xytext=(best_k, best_rmse + 0.15),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold', ha='center')


# Formatting
plt.title('Feature Selection: Validation RMSE vs. Number of Features (MRMR)', fontsize=14, pad=15)
plt.xlabel('Number of Top Features Kept (K)', fontsize=12)
plt.ylabel('Validation RMSE (Kelvin)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11, loc='upper left')

# Save and show
os.makedirs("Project/results", exist_ok=True)
save_path = "Project/results/mrmr_validation_curve.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved to: {save_path}")

name = f"Project/results/mrmr_validation_curve.png"
plt.savefig(name, dpi=300, bbox_inches="tight")
print(f"\nGraphic saved to: {name}")