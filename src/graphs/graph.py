import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
# Load the CSV file
# csv = "../csvs/model_results.csv"
csv = "../csvs/model_results_isolated.csv"

# df = pd.read_csv(csv)
df = pd.read_csv(csv, on_bad_lines='skip')


df['certainty'] = df['certainty'].astype(float)

model = "new_rawgat898"


if model == "xlsr_trained":
    # mult certainty by 100
    for i in df.index:
        y = min(df.loc[i, 'certainty'] * 200,1)
        df.loc[i, 'certainty'] = y

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
scalar = 1
plot_data['certainty'] = np.clip(plot_data['certainty'] * scalar, 0, 1)
# multiply certainty by 100, limit 1

# if cert > .4 set pred real
# for i in plot_data.index:
#     if plot_data.loc[i, 'certainty'] > .4:
#         plot_data.loc[i, 'pred-label'] = "Real"
#         plot_data.loc[i, 'certainty'] = .5
#     else:
#         plot_data.loc[i, 'pred-label'] = "Fake"
#         # plot_data.loc[i, 'certainty'] = .5

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
