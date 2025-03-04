import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backend_bases import NavigationToolbar2


i = 0
def setGraph(csv, ax, num):
    ax.clear()  # Clear the current plot
    df = pd.read_csv(csv)
    # headers file,correct-label,pred-label,certainty_real,certainty_fake


    # Ensure 'certainty' is parsed as float
    df['certainty_real'] = df['certainty_real'].astype(float)
    df['certainty_fake'] = df['certainty_fake'].astype(float)

    df = df.sort_values(by=['correct-label'], ascending=False)


    plot_data = pd.DataFrame()


    x_coordinates_real = []
    y_coordinates_real = []

    x_coordinates_fake = []
    y_coordinates_fake = []

    x_diff_coords = []
    y_diff_coords = []

    c_fake = len(df[df['correct-label'] == 'Fake'])
    c_real = len(df[df['correct-label'] == 'Real'])

    # count num correct label and pred label == 'Real'
    correct_real = len(df[(df['correct-label'] == 'Real') & (df['pred-label'] == 'Real')])
    correct_fake = len(df[(df['correct-label'] == 'Fake') & (df['pred-label'] == 'Fake')])

    end_real = 0

    tot_fake = 0
    tot_real = 0
    
    gap = 10

    y_fake_label = c_fake
    y_real_label = c_real + c_fake + gap

    mid = c_fake + gap/2

    for i in df.index:
        
        real = df.loc[i, 'certainty_real']
        fake = df.loc[i, 'certainty_fake']


        real = min(real,1)
        fake = min(fake,1)
        diff = abs(real - fake)

        if num == 12:
            real = min(real*1.25,1)
            fake = min(fake*500, 1)
            diff = abs(real-fake)
       
        correct_label = df.loc[i, 'correct-label']
        pred_label = df.loc[i, 'pred-label']


        


        tot_fake += 1 if correct_label == 'Fake' else 0
        tot_real += 1 if correct_label == 'Real' else 0



        y = y_fake_label if correct_label == 'Fake' else y_real_label

        y_coordinates_real.append(y)
        y_coordinates_fake.append(y)
        y_diff_coords.append(y)

        x_coordinates_real.append(real)
        x_coordinates_fake.append(fake)

        x_diff_coords.append(diff)

        y_fake_label -= 1 if correct_label == 'Fake' else 0
        y_real_label -= 1 if correct_label == 'Real' else 0


    # max y coordinate

    # Scatter plot using adjusted x and certainty as y
    ax.scatter(x_coordinates_real, y_coordinates_real, alpha=1, color='green', label='Real Certainty')
    ax.scatter(x_coordinates_fake, y_coordinates_fake, alpha=1, color='red', label='Fake Certainty')
    ax.scatter(x_diff_coords, y_diff_coords, alpha=1, color='blue', label='Difference')
    
    
    
    # Add titles and labels
    # ax.title(f'Scatter Plot of real vs fake certainty for Vocoder')
    ax.set_title(f'({num}): cor lab {csv.split("/")[-1]}')
    ax.set_xlabel('Category')
    ax.set_ylabel('Certainty')


    # set ticks every .1
    ax.set_xticks(np.arange(0, 1, .25))

    # set no y ticks
    ax.set_yticks([])

    # Set y-axis limits if needed (0 to 1)
    ax.set_xlim(0,1)


    # put label at end_real index of y axis as line

    ax.axhline(y=mid, color='black', linestyle='--')

    # label top half real on left edge out of graph
    ax.text(-.01, mid+c_real/2, f"Real: {correct_real}/{c_real}", fontsize=12, ha='right', va='center', color='green')

    # label bottom half fake on left edge out of graph
    ax.text(-.01, mid-c_fake/2, f"Fake: {correct_fake}/{c_fake}", fontsize=12, ha='right', va='center', color='red')

    # Show grid
    ax.grid()


    print()
    perc_fake_left = (c_fake-tot_fake)/c_fake*100
    perc_real_left = (c_real-tot_real)/c_real*100
    print(f"On board: fake - {100-perc_fake_left:.2f}% | real - {100-perc_real_left:.2f}%")
    print(f"delta: fake - {perc_fake_left:.2f}% | real - {perc_real_left:.2f}%")
    print()









home = NavigationToolbar2.home
   
# Load the CSV file
# csv = "../csvs/vocoder_real_fake.csv"
# csv = "../csvs/vocoder_real_fake_isolated.csv"

import os
# f = "../vocoder_train_csvs/"
f = "../vocoder_csvs/"
csvs = os.listdir(f)

def get_csv(i):
    return os.path.join(f, csvs[i])



fig, ax = plt.subplots(figsize=(10, 6)) 
def forward(self, *args, **kwargs):
    global i, ax

    if i == len(csvs) - 1:
        print("last file")
        return
    i+=1
    print (f"i: {i} file: {get_csv(i)}")

    setGraph(get_csv(i), ax, num=i)
    plt.draw()

def backward(self, *args, **kwargs):
    global i, ax
    if i == 0:
        print("first file")
        return
    i-=1
    print (f"i: {i} file: {get_csv(i)}")

    setGraph(get_csv(i), ax, num=i)
    plt.draw()


NavigationToolbar2.forward = forward
NavigationToolbar2.back = backward

# fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axes object

setGraph(get_csv(i), ax, num=i)  # Pass ax when calling setGraph
fig.set_size_inches(10, 6)

plt.show()