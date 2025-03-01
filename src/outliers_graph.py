import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('outliers.csv')

# Ensure 'certainty' is parsed as float
df['certainty'] = df['certainty'].astype(float)

# Create x-coordinates: every set of 4 rows gets the same x value.
x_coordinates = []
offset_x = 0
for i in range(len(df)):
    x_coordinates.append(offset_x)
    if i % 4 == 0:
        offset_x += 1

# Create a unique list of models, so that each model only appears once in the legend.
unique_models = df['model'].unique()

# Create a color map for different models
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_models)))
colors_map = dict(zip(unique_models, colors))

# Prepare a list to hold plotting data: each item is a tuple (x, y, color, model)
data = []
# Instead of using enumerate on df['model'], iterate with df.iterrows()
for i, row in df.iterrows():
    # Determine y value (category) based on correct and predicted labels.
    if row['correct-label'] == 'Real' and row['pred-label'] == 'Real':
        y = 1
    elif row['correct-label'] == 'Real' and row['pred-label'] == 'Fake':
        y = 2
    elif row['correct-label'] == 'Fake' and row['pred-label'] == 'Fake':
        y = 4
    elif row['correct-label'] == 'Fake' and row['pred-label'] == 'Real':
        y = 3
    
    y +=row['certainty']

    # Get the color based on the model
    model_name = row['model']
    color = colors_map[model_name]
    
    # Append the point data: x coordinate from x_coordinates list.
    data.append((x_coordinates[i], y, color, model_name))

# Plot the data without duplicating legend entries.
plt.figure(figsize=(10, 6))
models_plotted = set()  # Keeps track of models already added to legend

for point in data:
    x, y, color, model_name = point
    # Add label only the first time the model is plotted.
    if model_name not in models_plotted:
        plt.scatter(x, y, color=color, label=model_name)
        models_plotted.add(model_name)
    else:
        plt.scatter(x, y, color=color)

# Add titles and labels
plt.title('Scatter Plot of Certainty for Different Models')
plt.xlabel('Category Group')
plt.ylabel('Certainty Category')

# Set custom y-ticks to represent categories:
plt.yticks([1, 2, 3, 4], [
    'Real (Correct)',
    'Classified as Fake (Should be Real)',
    'Classified as Real (Should be Fake)',
    'Fake (Correct)',
])

# Adjust x-axis limits (adjust as needed)
plt.xlim(0, max(x_coordinates) + 1)
plt.ylim(0, 5)

plt.grid()

# Add a legend with a title
plt.legend(title='Models', loc='lower left', bbox_to_anchor=(0,0), framealpha=0.1)
plt.show()
