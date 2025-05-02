import json
from colorama import Fore, init
import sys

init(autoreset=True)



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

min_real = 7
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
        sep_label = res[model]["separated_results"]["label"]
        sep_pred = res[model]["separated_results"]["prediction"]
        og_label = res[model]["unseparated_results"]["label"]
        og_pred = res[model]["unseparated_results"]["prediction"]
        for label, pred in ((sep_label, sep_pred), (og_label, og_pred)):

            if label == "Real":
                votes_real += 1


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

"""
Using min_real: 7
Modified:
Real: 50/60 = 0.8333333333333334
Fake: 35/54 = 0.6481481481481481
Accuracy: 0.7407407407407407
"""
