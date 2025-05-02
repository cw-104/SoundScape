import json
from colorama import Fore, init
import sys

init(autoreset=True)


def process_real_label(model_name, pred, label, iso):
    """
    We make the bar for a model to classify as fake higher, this means that for a model to classify fake
    we are more confident it is correct.

    We want to reduce the number of false positives, so we make models lean towards real and only classify fake
    with high confidence.

    Returns true if the model result is real false if not
    """
    # we use predicted certainty value to shift towards real, meaning we cut-off certain values that if it were to guess fake, we change to real
    if "whisper" in model_name.lower():  # if low whisper pred count as fake
        if iso:
            pred *= 10 # iso lowers pred a lot
        if pred > 0.99 or pred < 0.01:
            return True
        return label == "Real"
    elif (
        "clad" in model_name.lower()
    ):  # clad score tends to be > .5 real < .5 fake (CLAD cert is where we directly calc real fake not necessarily the outputted label, and it operates differently, high is real, low is fake)
        if pred > 0.5:
            return True
    elif "xlsr" in model_name.lower():  # if prediction is very low, count as Real
        if not iso:
            pred *= 10 # iso lowers pred a lot
        if pred < 0.9:
            return True
        return label == "Real"

    elif "rawgat" in model_name.lower():
        if not iso:
            return pred > 0 # pretty good rates at just > 0
        if iso:
            # label for non-iso seems to be flipped and high certainty best
            return pred < 5
            
    elif "vocoder" in model.lower():
        if not iso:
            return label == "Real"
        if iso:
            if pred < .6:
                return True
            return label == "Real"



model_results = None
with open("eval_api_results.txt", "r") as f:
    model_results = json.loads(f.read())

all_results = []
with open("api_debug_all_results.txt", "r") as f:
    for line in f.readlines():
        if line.strip() == "":
            continue
        all_results.append(json.loads(line))
acc_res = {
    "combined": {
        "correct-real": 0,
        "total-real": 0,
        "correct-fake": 0,
        "total-fake": 0,
    }
}

min_real = 5
def print_res_of(model, acc_map):
    print(
        f"Real: {acc_map['correct-real']}/{acc_map['total-real']} = {acc_map['correct-real']/acc_map['total-real']}"
    )
    print(
        f"Fake: {acc_map['correct-fake']}/{acc_map['total-fake']} = {acc_map['correct-fake']/acc_map['total-fake']}"
    )
    print(
        f"Accuracy: {(acc_map['correct-real']/acc_map['total-real'] + acc_map['correct-fake']/acc_map['total-fake']) / 2}"
    )


for i, res in enumerate(all_results):
    votes_real = 0
    correct_label = model_results[i]["correct_label"]
    for model in res:
        if model == "filep":
            continue
        if model not in acc_res:
            acc_res[model] = {
                "og": {
                    "correct-real": 0,
                    "total-real": 0,
                    "correct-fake": 0,
                    "total-fake": 0,
                },
                "iso": {
                    "correct-real": 0,
                    "total-real": 0,
                    "correct-fake": 0,
                    "total-fake": 0,
                },
            }
        iso = True
        sep_label = res[model]["separated_results"]["label"]
        sep_pred = res[model]["separated_results"]["prediction"]
        og_label = res[model]["unseparated_results"]["label"]
        og_pred = res[model]["unseparated_results"]["prediction"]
        for label, pred in ((sep_label, sep_pred), (og_label, og_pred)):

            if label == "Real":
                votes_real += 1

            sub_key = "iso" if iso else "og"
            iso = False
            if correct_label == "Real":
                acc_res[model][sub_key]["total-real"] += 1
                if label == correct_label:
                    acc_res[model][sub_key]["correct-real"] += 1
            elif correct_label == "Fake":
                acc_res[model][sub_key]["total-fake"] += 1
                if label == correct_label:
                    acc_res[model][sub_key]["correct-fake"] += 1
    if votes_real >= min_real:
        guessed_label = "Real"
    else:
        guessed_label = "Fake"
    if correct_label == "Real":
        acc_res["combined"]["total-real"] += 1
        if guessed_label == correct_label:
            acc_res["combined"]["correct-real"] += 1
    elif correct_label == "Fake":
        acc_res["combined"]["total-fake"] += 1
        if guessed_label == correct_label:
            acc_res["combined"]["correct-fake"] += 1
for model in acc_res:
    if model == "combined":
        continue
    print(f"{Fore.BLUE}{model} og:")
    print_res_of(model, acc_res[model]["og"])
    print(f"{Fore.BLUE}{model} iso:")
    print_res_of(model, acc_res[model]["iso"])
    print()

print(f"{Fore.RED}Combined:")
print(
    f"Real: {acc_res['combined']['correct-real']}/{acc_res['combined']['total-real']} = {acc_res['combined']['correct-real']/acc_res['combined']['total-real']}"
)
print(
    f"Fake: {acc_res['combined']['correct-fake']}/{acc_res['combined']['total-fake']} = {acc_res['combined']['correct-fake']/acc_res['combined']['total-fake']}"
)
print(
    f"Accuracy: {(acc_res['combined']['correct-real']/acc_res['combined']['total-real'] + acc_res['combined']['correct-fake']/acc_res['combined']['total-fake']) / 2}"
)
print()

correct_real = 0
total_real = 0
correct_fake = 0
total_fake = 0

if len(sys.argv) > 1:
    min_real = int(sys.argv[1])
else:
    min_real = 6
print(f"Using min_real: {min_real}")
for i, res in enumerate(all_results):
    votes_real = 0
    correct_label = model_results[i]["correct_label"]

    for model in res:
        if model == "filep":
            continue
        iso = True
        sep_label = res[model]["separated_results"]["label"]
        sep_pred = res[model]["separated_results"]["prediction"]
        og_label = res[model]["unseparated_results"]["label"]
        og_pred = res[model]["unseparated_results"]["prediction"]
        for label, pred in ((sep_label, sep_pred), (og_label, og_pred)):
            if process_real_label(model, pred, label, iso):
                label = "Real"
            else:
                label = "Fake"

            if label == "Real":
                votes_real += 1
            iso = False

    if votes_real >= min_real:
        guessed_label = "Real"
    else:
        guessed_label = "Fake"

    if correct_label == "Real":
        total_real += 1
        if guessed_label == correct_label:
            correct_real += 1
            
    elif correct_label == "Fake":
        total_fake += 1
        if guessed_label == correct_label:
            correct_fake += 1
print(f"{Fore.GREEN}Modified:")
print(f"Real: {correct_real}/{total_real} = {correct_real/total_real}")
print(f"Fake: {correct_fake}/{total_fake} = {correct_fake/total_fake}")
print(f"Accuracy: {(correct_real/total_real + correct_fake/total_fake) / 2}")

"""
Using min_real: 7
Modified:
Real: 50/60 = 0.8333333333333334
Fake: 35/54 = 0.6481481481481481
Accuracy: 0.7407407407407407
"""