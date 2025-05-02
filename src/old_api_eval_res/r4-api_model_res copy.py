import json
from colorama import Fore, init
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
        if "whisper" in model_name.lower(): # if low whisper pred count as fake
            if iso:
                pred *= 10
            if pred > .99 or pred < .01:
                return True
            return label == "Real"
        elif "clad" in model_name.lower(): # clad score tends to be > .5 real < .5 fake (CLAD cert is where we directly calc real fake not necessarily the outputted label, and it operates differently, high is real, low is fake)
            if pred > .5:
                return True
        elif "xlsr" in model_name.lower(): # if prediction is very low, count as Real
            if not iso:
                pred *= 10
            if pred < .8:
                return True

        elif "rawgat" in model_name.lower():

            """
            === r4api_debug_all_results.txt and r4eval_api_results.txt ===
            ___________
            USING -- self.result_handler = DfResultHandler(-3, "Fake", "Real", 10, .45)
            res = self.model.evaluate_file(file_path)
            return min(.95,res.percent_certainty), res.classification
            ––––––––––––
            
            elif "rawgat" in model_name.lower():
                if not iso:
                    if pred < .45:
                        return False
                    return label == "Real"
                if iso and pred < .3:
                    return True
            results
            --------
            rawgat og:
                Real: 43/60 = 0.7166666666666667
                Fake: 22/55 = 0.4
                Accuracy: 0.5583333333333333
            rawgat iso:
                Real: 14/60 = 0.23333333333333334
                Fake: 48/55 = 0.8727272727272727
                Accuracy: 0.553030303030303

            ==
            SIMPLE FLIP
            ==
            return label != "Real"
            results
            --------
            rawgat og:
            Real: 0/60 = 0.0
            Fake: 54/55 = 0.9818181818181818
            Accuracy: 0.4909090909090909
                rawgat iso:
            Real: 32/60 = 0.5333333333333333
            Fake: 28/55 = 0.509090909090909
            Accuracy: 0.5212121212121212
            """
            if not iso:
                if pred < .45:
                    return False
                return label == "Real"
            if iso and pred < .3:
                return True
        # leave vocoder, it does not have strong outliers, solid as is

        # This is for if(multi[0] > multi[1]): "Real" > if use this, needs to be flipped
        elif "vocoder" in model.lower():
            """
            === r4api_debug_all_results.txt and r4eval_api_results.txt ===
            
            FLIP -- MULTI -- multi[1] > multi[0] = REAL 
            pred = multi[0/1]
            label = "Fake" if label == "Real" else "Real" # Flip to multi[1] > multi[0]
            return label == "Real"

            results
            --------
            vocoder og:
                Real: 49/60 = 0.8166666666666667
                Fake: 24/55 = 0.43636363636363634
                Accuracy: 0.6265151515151515
            vocoder iso:
                Real: 0/60 = 0.0
                Fake: 55/55 = 1.0
                Accuracy: 0.5

            ===
            if not iso:
                label = "Fake" if label == "Real" else "Real" # Flip to multi[1] > multi[0]
                return label == "Real"

            results
            --------
            if iso:
                if pred > .7:
                    return False
                return label == "Real"
            vocoder og:
                Real: 49/60 = 0.8166666666666667
                Fake: 24/55 = 0.43636363636363634
                Accuracy: 0.6265151515151515
            vocoder iso:
                Real: 54/60 = 0.9
                Fake: 12/55 = 0.21818181818181817
                Accuracy: 0.5590909090909091
            """
            if not iso:
                label = "Fake" if label == "Real" else "Real" # Flip to multi[1] > multi[0]
                return label == "Real"
            if iso:
                if pred > .7:
                    return False
                return label == "Real"

        return False
            


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
    'combined': {
        'correct-real': 0,
        'total-real': 0,
        'correct-fake': 0,
        'total-fake': 0,
    }
}
min_real = 5
def print_res_of(model, acc_map):
    print(f"Real: {acc_map['correct-real']}/{acc_map['total-real']} = {acc_map['correct-real']/acc_map['total-real']}")
    print(f"Fake: {acc_map['correct-fake']}/{acc_map['total-fake']} = {acc_map['correct-fake']/acc_map['total-fake']}")
    print(f"Accuracy: {(acc_map['correct-real']/acc_map['total-real'] + acc_map['correct-fake']/acc_map['total-fake']) / 2}")
for i, res in enumerate(all_results):
    votes_real = 0
    correct_label = model_results[i]['correct_label']
    for model in res:
        if model == "filep": continue
        if model not in acc_res:
            acc_res[model] = {
                'og': {
                'correct-real': 0,
                'total-real': 0,
                'correct-fake': 0,
                'total-fake': 0,
            },
            'iso': {
                'correct-real': 0,
                'total-real': 0,
                'correct-fake': 0,
                'total-fake': 0,
            }
        }
        iso = True
        sep_label = res[model]["separated_results"]['label']
        sep_pred = res[model]["separated_results"]['prediction']
        og_label = res[model]["unseparated_results"]['label']
        og_pred = res[model]["unseparated_results"]['prediction']
        for label, pred in ((sep_label, sep_pred), (og_label, og_pred)):
            
            # if label == "Real":
            #     votes_real+=1
            if process_real_label(model, pred, label, iso):
                label = "Real"
                votes_real+=1
            else:
                label = "Fake"

            sub_key = 'iso' if iso else 'og'
            iso = False
            if correct_label == "Real":
                acc_res[model][sub_key]['total-real']+=1
                if label == correct_label:
                    acc_res[model][sub_key]['correct-real']+= 1
            elif correct_label ==  "Fake":
                acc_res[model][sub_key]['total-fake']+=1
                if label == correct_label:
                    acc_res[model][sub_key]['correct-fake']+=1
    if votes_real >= min_real:
        guessed_label = "Real"
    else:
        guessed_label = "Fake"
    if correct_label == "Real":
        acc_res['combined']['total-real']+=1
        if guessed_label == correct_label:
            acc_res['combined']['correct-real']+= 1
    elif correct_label ==  "Fake":
        acc_res['combined']['total-fake']+=1
        if guessed_label == correct_label:
            acc_res['combined']['correct-fake']+=1
for model in acc_res:
    if model == "combined": continue
    print(f"{Fore.BLUE}{model} og:")
    print_res_of(model, acc_res[model]['og'])
    print(f"{Fore.BLUE}{model} iso:")
    print_res_of(model, acc_res[model]['iso'])
    print()

print(f"{Fore.RED}Combined:")
print(f"Real: {acc_res['combined']['correct-real']}/{acc_res['combined']['total-real']} = {acc_res['combined']['correct-real']/acc_res['combined']['total-real']}")
print(f"Fake: {acc_res['combined']['correct-fake']}/{acc_res['combined']['total-fake']} = {acc_res['combined']['correct-fake']/acc_res['combined']['total-fake']}")
print(f"Accuracy: {(acc_res['combined']['correct-real']/acc_res['combined']['total-real'] + acc_res['combined']['correct-fake']/acc_res['combined']['total-fake']) / 2}")
print()

correct_real = 0
total_real = 0
correct_fake = 0
total_fake = 0

min_real = 6
for i, res in enumerate(all_results):
    votes_real = 0
    correct_label = model_results[i]['correct_label']

    for model in res:
        # if "xlsr" not in model.lower(): continue
        if model == "filep": continue
        iso = True
        sep_label = res[model]["separated_results"]['label']
        sep_pred = res[model]["separated_results"]['prediction']
        og_label = res[model]["unseparated_results"]['label']
        og_pred = res[model]["unseparated_results"]['prediction']
        for label, pred in ((sep_label, sep_pred), (og_label, og_pred)):
            if process_real_label(model, pred, label, iso):
                label = "Real"
            else:
                label = "Fake"
            # if "whisper" in model.lower():
            #     if not iso and pred < .25:
            #         label = "Fake"
            # if "clad" in model.lower():
            #     if pred > .5:
            #         label = "Real"


            # if "rawgat" in model.lower():
            #     if not iso and pred < 0.025:
            #         label = "Fake"
            #     if iso:
            #         if pred > .15:
            #             label = "Real"
            #         else:
            #             label = "Fake"

            # if "vocoder" in model.lower():
            #     if iso and pred > .2:
            #         label = "Real"
            #     if not iso and pred > .25:
            #         label = "Real"

            # if "xlsr" in model.lower():
            #     if not iso and pred < 0.25:
            #         label = "Fake"

                # if not iso and pred < 1:
                #     label = "Real"
                # if iso and pred < 1.25: # .5 for new bindings
                #     label = "Real"

            if label == "Real":
                votes_real+=1
            iso = False
    # print(votes_real)
    if votes_real >= min_real:
        guessed_label = "Real"
    else:
        guessed_label = "Fake"
    
    if correct_label == "Real":
        total_real+=1
        if guessed_label == correct_label:
            correct_real+= 1
    elif correct_label ==  "Fake":
        total_fake+=1
        if guessed_label == correct_label:
            correct_fake+=1
print(f"{Fore.GREEN}Modified:")
print(f"Real: {correct_real}/{total_real} = {correct_real/total_real}")
print(f"Fake: {correct_fake}/{total_fake} = {correct_fake/total_fake}")
print(f"Accuracy: {(correct_real/total_real + correct_fake/total_fake) / 2}")


"""

CLAD og:
Real: 42/60 = 0.7
Fake: 40/55 = 0.7272727272727273
Accuracy: 0.7136363636363636
CLAD iso:
Real: 42/60 = 0.7
Fake: 40/55 = 0.7272727272727273
Accuracy: 0.7136363636363636

whisper_specrnet og:
Real: 43/60 = 0.7166666666666667
Fake: 20/55 = 0.36363636363636365
Accuracy: 0.5401515151515152
whisper_specrnet iso:
Real: 35/60 = 0.5833333333333334
Fake: 23/55 = 0.41818181818181815
Accuracy: 0.5007575757575757

xlsr og:
Real: 48/60 = 0.8
Fake: 24/55 = 0.43636363636363634
Accuracy: 0.6181818181818182
xlsr iso:
Real: 55/60 = 0.9166666666666666
Fake: 9/55 = 0.16363636363636364
Accuracy: 0.5401515151515152

rawgat og:
Real: 43/60 = 0.7166666666666667
Fake: 22/55 = 0.4
Accuracy: 0.5583333333333333
rawgat iso:
Real: 14/60 = 0.23333333333333334
Fake: 48/55 = 0.8727272727272727
Accuracy: 0.553030303030303

vocoder og:
Real: 49/60 = 0.8166666666666667
Fake: 24/55 = 0.43636363636363634
Accuracy: 0.6265151515151515
vocoder iso:
Real: 54/60 = 0.9
Fake: 12/55 = 0.21818181818181817
Accuracy: 0.5590909090909091

Combined:
Real: 55/60 = 0.9166666666666666
Fake: 17/55 = 0.3090909090909091
Accuracy: 0.6128787878787878

–––––––––––– min_real = 6
Modified:
Real: 49/60 = 0.8166666666666667
Fake: 35/55 = 0.6363636363636364
Accuracy: 0.7265151515151516
––––––––––––
"""
