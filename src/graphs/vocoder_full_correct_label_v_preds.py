import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.widgets import CheckButtons

header = ['file', 'correct-label', 'pred-label', 'certainty_real', 'certainty_fake', 'gt', 'wavegrad', 'diffwave', 'parallel wave gan', 'wavernn', 'wavenet', 'melgan', "custom"]
colors = ['green', 'red', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', "yellow"]

# arrays of booleans whether should be active or not
active = [False for _ in header[3:]]
active[9] = True
# Directory with CSV files (update path as needed)
f = "../vocoder_csvs/"
csvs = os.listdir(f)
csvs.sort()  # Optional: sort filenames

# Global index to track current CSV file and scatter objects
i = 9
scatters = []

def get_csv(idx):
    return os.path.join(f, csvs[idx])

skip_correct_gt = False
skip_incorrect_gt = False

skip_incorrect_cert = False
skip_correct_cert = False

def setGraph(csv_path, ax, num=0):
    print(f"Setting graph for {csv_path}")
    global scatters
    scatters = []  # reset scatter list
    df = pd.read_csv(csv_path)
    # Ensure certainty values are floats
    for col in header[3:-1]:
        df[col] = df[col].astype(float)
    
    # sort by correct-label: assuming you want Reals first or Fakes first based on your logic
    df = df.sort_values(by=['correct-label'], ascending=False)

    tot_reals = len(df[df['correct-label'] == 'Real'])
    tot_fakes = len(df[df['correct-label'] == 'Fake'])

    gap = 10
    real_y = tot_reals + tot_fakes + gap
    fake_y = 0

    # Prepare empty lists for plotting data for each method
    plot_y = [[] for _ in header[3:]]
    plot_x = [[] for _ in header[3:]]
    
    ax.clear()  # Clear previous plot

    gtwv_correct_real = 0
    gtwv_correct_fake = 0

    cert_correct_real = 0
    cert_correct_fake = 0

    # Loop over each row (use idx to avoid shadowing global i)
    for idx in df.index:
        is_gt_correct = False
        is_cert_correct = False
        correct_label = df.loc[idx, 'correct-label']
        real_cert = df.loc[idx, 'certainty_real']
        fake_cert = df.loc[idx, 'certainty_fake']
        gt = df.loc[idx, 'gt']
        wavegrad = df.loc[idx, 'wavegrad']

        gtwv_pred = "Real" if gt>wavegrad else "Fake"
        if gtwv_pred == correct_label:
            gtwv_correct_real += 1 if gtwv_pred == 'Real' else 0
            gtwv_correct_fake += 1 if gtwv_pred == 'Fake' else 0
            is_gt_correct = True

        cert_pred = "Real" if real_cert>fake_cert else "Fake"
        if cert_pred == correct_label:
            cert_correct_real += 1 if cert_pred == 'Real' else 0
            cert_correct_fake += 1 if cert_pred == 'Fake' else 0
            is_cert_correct = True
        

        if not ((is_cert_correct and skip_correct_cert) or (not is_cert_correct and skip_incorrect_cert) or (is_gt_correct and skip_correct_gt) or (not is_gt_correct and skip_incorrect_gt)):
            for j, col in enumerate(header[3:-1]):
                data = df.loc[idx, col]
                if j != 9 and j >= 4:
                    data = min(data*1000, 1)
                plot_x[j].append(data)
                plot_y[j].append(fake_y if correct_label == 'Fake' else real_y)
            fake_y += 1 if correct_label == 'Fake' else 0
            real_y -= 1 if correct_label == 'Real' else 0

            # neg
            diffwave = min(df.loc[idx, 'diffwave'] * 1000,1)
            melgan = min(df.loc[idx, 'melgan'] * 1000,1)
            wavenet = min(df.loc[idx, 'wavenet'] * 1000,1)
            wavernn = min(df.loc[idx, 'wavernn'] * 1000,1)
            wave_gan = min(df.loc[idx, 'parallel wave gan'] * 1000,1)
            diffwave = min(df.loc[idx, 'diffwave'] * 1000,1)
            wavegrad = min(df.loc[idx, 'wavegrad'],1)
            fake_cert = min(df.loc[idx, 'certainty_fake'],1)
            real_cert = min(df.loc[idx, 'certainty_real'],1)

            gt = gt - .75
            melgan = -max(.6-melgan, 0) * 2
            wavernn = -max(.4-wavernn, 0)
            wave_gan = -max(.4-wave_gan, 0)
            diffwave = -max(.4-diffwave, 0) * 1.5
            wavegrad = wavegrad - .25
            wavenet = -max(.2-wavenet, 0)



            custom = fake_cert-real_cert + melgan + wavernn + wave_gan + wavenet + diffwave + gt + wavegrad
            custom = min(max(custom, 0), 1)
            plot_x[9].append(abs(custom))
            plot_y[9].append(fake_y if correct_label == 'Fake' else real_y)

    # Plot each scatter and store the object for toggling
    for j, col in enumerate(header[3:]):
        sc = ax.scatter(plot_x[j], plot_y[j], label=col, color=colors[j])
        scatters.append(sc)



    # draw line to separate real and fake
    ax.axhline(tot_fakes + gap/2, color='black', linestyle='--')
    # add label top section and bottom section to the left of the graph (not overlapping with the graph) colored large bold
    ax.text(-0.25, 75, 'Real', color='Green', fontsize=15, fontweight='bold')
    # below, cert acc and gtwv acc
    t = f"gt/wav: {gtwv_correct_real}/{tot_reals} ({gtwv_correct_real/tot_reals*100:.2f}%)" + "\n"
    t += f"cert: {cert_correct_real}/{tot_reals} ({cert_correct_real/tot_reals*100:.2f}%)"
    ax.text(-0.3, 65, t, color='black', fontsize=8)

    ax.text(-0.25, 15, 'Fake', color='Red', fontsize=15, fontweight='bold')
    # below, cert acc and gtwv acc
    t = f"gt/wav: {gtwv_correct_fake}/{tot_fakes} ({gtwv_correct_fake/tot_fakes*100:.2f}%)" + "\n"
    t += f"cert: {cert_correct_fake}/{tot_fakes} ({cert_correct_fake/tot_fakes*100:.2f}%)"
    ax.text(-0.3, 10, t, color='black', fontsize=8)

    ax.set_title(f"File {num}: {os.path.basename(csv_path)}")
    ax.set_xlabel('Index')
    ax.set_xlim(-0.1, 1.1)

    ax.set_ylabel('Certainty')
    for j, sc in enumerate(scatters):
        sc.set_visible(active[j])

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
setGraph(get_csv(i), ax, num=i)

# Define custom forward/backward functions
def forward(self, *args, **kwargs):
    global i, ax
    i += 1
    if i >= len(csvs):
        i = len(csvs) - 1
        print("Last file reached")
        return
    print(f"Forward: i = {i}, file: {get_csv(i)}")
    setGraph(get_csv(i), ax, num=i)
    plt.draw()

def backward(self, *args, **kwargs):
    global i, ax
    i -= 1
    if i < 0:
        i = 0
        print("First file reached")
        return
    print(f"Backward: i = {i}, file: {get_csv(i)}")
    setGraph(get_csv(i), ax, num=i)
    plt.draw()

# Override the NavigationToolbar2 forward and back functions
NavigationToolbar2.forward = forward
NavigationToolbar2.back = backward

def toggle_visibility(label):
    index = header[3:].index(label)
    active[index] = not active[index]
    for j, sc in enumerate(scatters):
        sc.set_visible(active[j])
    plt.draw()

# Adjust the subplot to leave space for the checkboxes
plt.subplots_adjust(right=0.8)
plt.subplots_adjust(left=.15)
# Increase the height of the check box axes for more vertical space
check_ax = fig.add_axes([0.8, 0.2, 0.18, 0.6], frameon=False)


check = CheckButtons(check_ax, header[3:], active)
check.on_clicked(toggle_visibility)
for i, label in enumerate(check.labels):
    label.set_color(colors[i])
# make check box bigger

skip_toggles = ['Skip Correct GT', 'Skip Incorrect GT', 'Skip Incorrect Cert', 'Skip Correct Cert']
def toggle_skip(label):
    print(label)
    global skip_correct_gt, skip_incorrect_gt, skip_incorrect_cert, skip_correct_cert
    if label == skip_toggles[0]:
        skip_correct_gt = not skip_correct_gt
    elif label == skip_toggles[1]:
        skip_incorrect_gt = not skip_incorrect_gt
    elif label == skip_toggles[2]:
        skip_incorrect_cert = not skip_incorrect_cert
    elif label == skip_toggles[3]:
        skip_correct_cert = not skip_correct_cert
    setGraph(get_csv(i), ax, num=i)
    plt.draw()

check_skip = CheckButtons(fig.add_axes([0.8, 0.8, 0.18, 0.1], frameon=False), skip_toggles, [False, False, False, False])
check_skip.on_clicked(toggle_skip)

fig.set_size_inches(10, 6)
plt.show()
