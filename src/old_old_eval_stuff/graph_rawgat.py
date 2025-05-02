from final_val_res_all_models import get_results
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Assuming you have already defined results and model_paths
results = get_results(model_kinds=['rawgat'])
model_paths = list(results.keys())  # Convert keys to a list for easier access

# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2, top=0.8)  # Adjust the bottom and top to make space for buttons

# Initialize the isolated state
isolated = False

# Function to update the plot based on the selected model path and isolated state
def update_plot(model_path):
    ax.clear()  # Clear the current axes

    type = 'iso' if isolated else 'og'
    model_raw_data = results[model_path][type]['raw']

    # Extracting the data for plotting
    x = []
    y = []
    for entry in model_raw_data:
        filep, certainty, label, correct_label = entry
        x.append(correct_label)  # 0 for Real, 1 for Fake
        y.append(certainty)      # Certainty value
    ax.set_ylim(min(y)-.1, max(y)+.1)  # Sets the y-axis limits from 0 to 1
    # Create the scatter plot
    ax.scatter(x, y, alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Real', 'Fake'])
    ax.set_xlabel('Correct Label')
    ax.set_ylabel('Certainty')
    ax.set_title(f'Certainty vs Correct Label for {model_path} (Isolated: {isolated})')
    ax.grid()
    plt.draw()  # Redraw the figure

# Create buttons for each model path
buttons = []
for i, model_path in enumerate(model_paths):
    ax_button = plt.axes([i * 0.15, 0.05, 0.1, 0.075])  # Position of the button
    button = Button(ax_button, model_path.split('/')[-1])  # Use the model name as the button label
    button.on_clicked(lambda event, path=model_path: update_plot(path))  # Update plot on click
    buttons.append(button)

# Toggle button for isolated state at the top center
ax_toggle = plt.axes([0.4, 0.85, 0.2, 0.075])  # Position of the toggle button (top center)
toggle_button = Button(ax_toggle, 'Toggle Isolated')

def toggle_isolated(event):
    global isolated
    isolated = not isolated  # Toggle the state
    update_plot(model_paths[0])  # Update the plot with the new state

toggle_button.on_clicked(toggle_isolated)  # Link the toggle button to the function

# Initial plot
update_plot(model_paths[0])  # Plot the first model path initially

# Show the plot
plt.show()
