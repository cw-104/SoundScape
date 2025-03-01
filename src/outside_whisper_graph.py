import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('outside_whisper_range.csv')

# Ensure 'certainty' is parsed as float
df['certainty'] = df['certainty'].astype(float)

# Get unique models
unique_models = df['model'].unique()

# Create a color map for different models
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_models)))

# Create a figure for the plot
plt.figure(figsize=(10, 6))

# Iterate through each model and plot
for index, (color, model) in enumerate(zip(colors, unique_models)):
    # Filter the DataFrame for the specified model
    filtered_df = df[df['model'] == model]

    # Apply transformations for specific models if needed
    if model == "xlsr-finetuned":
        for i in filtered_df.index:
            y = filtered_df.loc[i, 'certainty'] * 1000 * 2
            y = np.clip(y ** 3, 0, 1) - .1
            filtered_df.loc[i, 'certainty'] = y

    # Prepare data for plotting
    plot_data = pd.DataFrame()
    plot_data['certainty'] = filtered_df['certainty']

    # Adjust x-coordinates based on the conditions
    plot_data['x'] = 0  # Default x-coordinate
    plot_data.loc[(filtered_df['correct-label'] == 'Real') & (filtered_df['pred-label'] == 'Real'), 'x'] = 1  # Real (Correct)
    plot_data.loc[(filtered_df['correct-label'] == 'Fake') & (filtered_df['pred-label'] == 'Real'), 'x'] = 2  # Classified as real (Should be Fake)
    plot_data.loc[(filtered_df['correct-label'] == 'Fake') & (filtered_df['pred-label'] == 'Fake'), 'x'] = 3  # Fake (Correct)
    plot_data.loc[(filtered_df['correct-label'] == 'Real') & (filtered_df['pred-label'] == 'Fake'), 'x'] = 4  # Classified as fake (Should be Real)

    # Offset the x-coordinates by 0.1 for each model
    plot_data['x'] += index * 0.1

    # Scatter plot for the current model
    plt.scatter(plot_data['x'], plot_data['certainty'], alpha=0.7, color=color, label=model)

# Add titles and labels
plt.title('Scatter Plot of Certainty for Different Models')
plt.xlabel('Category')
plt.ylabel('Certainty')

# Set x-ticks to represent categories, adjusting for the offset
plt.xticks([1 + 0.1 * (len(unique_models) - 1) / 2, 2 + 0.1 * (len(unique_models) - 1) / 2, 
            3 + 0.1 * (len(unique_models) - 1) / 2, 4 + 0.1 * (len(unique_models) - 1) / 2], 
           ['Real (Correct)', 'Classified as Real (Should be Fake)', 'Fake (Correct)', 'Classified as Fake (Should be Real)'])

# Set y-axis limits if needed (0 to 1)
plt.ylim(0, 1)

# Show grid
plt.grid()

# Add a legend
plt.legend(title='Models')

# Show the plot
plt.show()
