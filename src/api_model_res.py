import json

model_results = None
with open("eval_api_results.txt", "r") as f:
    model_results = json.loads(f.read())


all_results = []
with open("api_debug_all_results.txt", "r") as f:
    for line in f.readlines():
        if line.strip() == "":
            continue
        all_results.append(json.loads(line))

correct_real = 0
total_real = 0
correct_fake = 0
total_fake = 0
min_real = 3
for i, res in enumerate(all_results):
    if i > len(model_results) - 1:
        print("res n > ", len(model_results))
        break
    votes_real = 0
    correct_label = model_results[i]['correct_label']
    for model in res:
        iso = True
        for label, pred in ((res[model]["unseparated_results"]['label'], res[model]["unseparated_results"]['prediction']), (res[model]["separated_results"]['label'], res[model]
        ["separated_results"]['prediction'])):
            if "whisper" in model.lower():
                pass
            
            if label == "Real":
                votes_real+=1
            iso = False

    
    print(votes_real)
    if votes_real > min_real:
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
print(f"Real: {correct_real}/{total_real} = {correct_real/total_real}")
print(f"Fake: {correct_fake}/{total_fake} = {correct_fake/total_fake}")
print(f"Accuracy: {(correct_real/total_real + correct_fake/total_fake) / 2}")

