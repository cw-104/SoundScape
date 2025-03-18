import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import NavigationToolbar2

# Load the CSV file (adjust the path as needed)
csv_path = "../csvs/model_results_isolated.csv"
# csv_path = "../csvs/model_results.csv"
df = pd.read_csv(csv_path, on_bad_lines='skip')
df['certainty'] = df['certainty'].astype(float)

main_model = "xlsr"
comparison_models = ['xlsr_epoch_86.pth', '57_xlsr_epoch20', 'other_xlsr_76', 'other_xlsr_12']

# Scaling factors (if you want to adjust the certainty values)
sc_main = 200
sc_comp = 100

# Global index for the current comparison model
i = 0

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

def update_plot():
    """Clear the axes and redraw the scatter plot for main_model vs. current comparison model."""
    global i, ax, df, main_model, comparison_models
    ax.clear()
    
    current_model = comparison_models[i]
    
    # --- Plot main model data (blue) ---
    filtered_main = df[df['model'] == main_model].copy()
    filtered_main['x'] = 0
    filtered_main.loc[filtered_main['correct-label'] == 'Real', 'x'] = 1 - 0.05  # slight left offset
    filtered_main.loc[filtered_main['correct-label'] == 'Fake', 'x'] = 2 - 0.05  # slight left offset
    if sc_main > 1:
        filtered_main['certainty'] = filtered_main['certainty'].apply(lambda y: min(y * sc_main, 1))
    ax.scatter(filtered_main['x'], filtered_main['certainty'], alpha=1, color="green", label=main_model)
    
    # --- Plot comparison model data (green) ---
    filtered_comp = df[df['model'] == current_model].copy()
    filtered_comp['x'] = 0
    filtered_comp.loc[filtered_comp['correct-label'] == 'Real', 'x'] = 1 + 0.05  # slight right offset
    filtered_comp.loc[filtered_comp['correct-label'] == 'Fake', 'x'] = 2 + 0.05  # slight right offset
    if sc_comp > 1:
        filtered_comp['certainty'] = filtered_comp['certainty'].apply(lambda y: min(y * sc_comp, 1))
    ax.scatter(filtered_comp['x'], filtered_comp['certainty'], alpha=1, color="blue", label=current_model)
    
    # --- Set up titles, labels, and grid ---
    ax.set_title(f'{main_model} vs {current_model} Certainty Comparison')
    ax.set_xlabel('Category')
    ax.set_ylabel('Certainty')
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['', 'Real Label', 'Fake Label', '', ''])
    ax.set_ylim(0, 1)

    # y ticks interval .25
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(True)
    ax.legend()
    
    # Redraw the canvas
    fig.canvas.draw()

def forward(self, *args, **kwargs):
    """Advance to the next comparison model using the toolbar's forward button."""
    global i, comparison_models
    if i >= len(comparison_models) - 1:
        print("Already at last comparison model.")
        return
    i += 1
    print(f"Switching to model index {i}: {comparison_models[i]}")
    update_plot()

def back(self, *args, **kwargs):
    """Go back to the previous comparison model using the toolbar's back button."""
    global i, comparison_models
    if i <= 0:
        print("Already at first comparison model.")
        return
    i -= 1
    print(f"Switching to model index {i}: {comparison_models[i]}")
    update_plot()

# Override the NavigationToolbar2 methods
NavigationToolbar2.forward = forward
NavigationToolbar2.back = back

# Draw the initial plot
update_plot()
plt.show()
