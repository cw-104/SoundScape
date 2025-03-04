import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import os

i = 0

def setGraph(csv,ax, num=0):
    ax.clear()  # Clear the current plot

    # set size of plot
    df = pd.read_csv(csv)

    # Ensure 'certainty' is parsed as float
    df['certainty_real'] = df['certainty_real'].astype(float)
    df['certainty_fake'] = df['certainty_fake'].astype(float)



    plot_data = pd.DataFrame()


    x_coordinates_real = []
    y_coordinates_real = []

    x_coordinates_fake = []
    y_coordinates_fake = []

    x_coordinates_diff = []
    y_coordinates_diff = []


    total_fake_labels = 0
    total_real_labels = 0

    correct_fake = 0
    correct_real = 0

    incorrect_fake = 0
    incorrect_real = 0


    # sort by correct label 
    df = df.sort_values(by=['correct-label'], ascending=True)

    for i in df.index:

        real = df.loc[i, 'certainty_real']
        fake = df.loc[i, 'certainty_fake']
        
        real = min(real,1)
        fake = min(fake,1)
        diff = min(abs(real - fake),1)
        
        correct_label = df.loc[i, 'correct-label']
        pred_label = df.loc[i, 'pred-label']




        # if num == 12:
        #     real = min(real*1.25,1)
        #     fake = min(fake*1000, 1)
        
        # if num == 5:
        #     if real < .4:
        #         pred_label = "Fake"
        #     if diff > .4:
        #         pred_label = "Fake"
        #     if fake > .5:
        #         pred_label = "Fake"

        #     if diff < .1:
        #         pred_label = "Real"

        # if num == 14:
        #     if real < .55:
        #         pred_label = "Fake"
        #     if diff < .1 or diff > .5:
        #         pred_label = "Fake"
        #     if fake > .45:
        #         pred_label = "Fake"
                




            
        total_real_labels += 1 if correct_label == 'Real' else 0
        total_fake_labels += 1 if correct_label == 'Fake' else 0

        
        if correct_label == 'Real' and pred_label == 'Real': # Real (pred Real)
            x_offset = 1
            correct_real += 1
        elif correct_label == 'Real' and pred_label == 'Fake': # Real (pred Fake)
            x_offset = 2
            incorrect_real += 1
        elif correct_label == 'Fake' and pred_label == 'Fake': # Fake (pred fake)
            x_offset = 3
            correct_fake += 1
        elif correct_label == 'Fake' and pred_label == 'Real': # Fake (pred real)
            x_offset = 4
            incorrect_fake += 1


        y_coordinates_real.append(i)
        y_coordinates_fake.append(i)
        y_coordinates_diff.append(i)

        x_coordinates_real.append(real + x_offset)
        x_coordinates_fake.append(fake + x_offset)
        x_coordinates_diff.append(diff + x_offset)






    # percentages
    correct_real_acc = correct_real / total_real_labels
    correct_fake_acc = correct_fake / total_fake_labels
    # EER
    EER = (incorrect_fake / total_fake_labels + incorrect_real / total_real_labels)/2

    # print accuracy
    print(f"Correct Real: {correct_real}/{total_real_labels} | {correct_real_acc*100:.2f}%")
    print(f"Correct Fake: {correct_fake}/{total_fake_labels} | {correct_fake_acc*100:.2f}%")
    print(f"EER: {EER*100:.2f}%")


    # Scatter plot using adjusted x and certainty as y
    ax.scatter(x_coordinates_real, y_coordinates_real, alpha=1, color='green', label='Real Certainty')
    ax.scatter(x_coordinates_fake, y_coordinates_fake, alpha=1, color='red', label='Fake Certainty')
    ax.scatter(x_coordinates_diff, y_coordinates_diff, alpha=1, color='blue', label='Difference')

    # Add titles and labels
    # ax.title(f'vocoder r v f: {csv}')
    ax.set_title(f"({num}): rvf {csv.split('/')[-1]}")
    ax.set_xlabel('Category')
    ax.set_ylabel('Certainty')

    # Set x-ticks to represent categories
    ax.set_xticks([1, 2, 3, 4], ['Real (Correct)', 'Real (Pred Fake)', 'Fake (Correct)', 'Fake (Pred Real)'])

    # Set y-axis limits if needed (0 to 1)
    ax.set_xlim(1, 5)

    # Show grid
    ax.grid()


import os
f = "../vocoder_train_csvs/"
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


from matplotlib.backend_bases import NavigationToolbar2

NavigationToolbar2.forward = forward
NavigationToolbar2.back = backward

# fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axes object

setGraph(get_csv(i), ax, num=i)  # Pass ax when calling setGraph
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.show()