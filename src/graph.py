import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
# Load the CSV file
df = pd.read_csv('model_results.csv')

# Ensure 'certainty' is parsed as float
df['certainty'] = df['certainty'].astype(float)

# Specify the model you want to analyze
# model = "xlsr-finetuned"  # Change this to the model you want to analyze
model = "whisper_specrnet"

# Filter the DataFrame for the specified model
filtered_df = df[df['model'] == model]

if model == "xlsr-finetuned":
    for i in filtered_df.index:
        y = filtered_df.loc[i, 'certainty'] * 1000*2
        y = np.clip(y ** 3, 0, 1) - .1
        filtered_df.loc[i, 'certainty'] = y


# for i in filtered_df.index:
#     if filtered_df.loc[i, 'pred-label'] == 'Fake':
#         if filtered_df.loc[i, 'certainty'] - .25 > 0:
#             filtered_df.loc[i, 'pred-label'] = "Real"



plot_data = pd.DataFrame()
plot_data['certainty'] = filtered_df['certainty']

# Adjust x-coordinates based on the conditions
plot_data['x'] = 0  # Default x-coordinate
plot_data.loc[(filtered_df['correct-label'] == 'Real') & (filtered_df['pred-label'] == 'Real'), 'x'] = 1  # Real (Correct)
plot_data.loc[(filtered_df['correct-label'] == 'Fake') & (filtered_df['pred-label'] == 'Real'), 'x'] = 2  # Classified as real (Should be Fake)
plot_data.loc[(filtered_df['correct-label'] == 'Fake') & (filtered_df['pred-label'] == 'Fake'), 'x'] = 3  # Fake (Correct)
plot_data.loc[(filtered_df['correct-label'] == 'Real') & (filtered_df['pred-label'] == 'Fake'), 'x'] = 4  # Classified as fake (Should be Real)

# Now you can create a scatter plot using the adjusted coordinates
plt.figure(figsize=(10, 6))

# Scatter plot using adjusted x and certainty as y
plt.scatter(plot_data['x'], plot_data['certainty'], alpha=1)

# Add titles and labels
plt.title(f'Scatter Plot of Certainty for Model: {model}')
plt.xlabel('Category')
plt.ylabel('Certainty')

# Set x-ticks to represent categories
plt.xticks([1, 2, 3, 4], ['Real (Correct)', 'Classified as Real (Should be Fake)', 'Fake (Correct)', 'Classified as Fake (Should be Real)'])

# Set y-axis limits if needed (0 to 1)
plt.ylim(0, 1)

# Show grid
plt.grid()

# Show the plot
plt.show()
