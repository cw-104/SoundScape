import os
from backend.Evaluate import DeepfakeClassificationModel
from backend.Results import DfResultHandler


def print_res(res1, res2, str, lower : bool):
    print()
    res1.sort(key=lambda x: x.file_name)
    res2.sort(key=lambda x: x.file_name)
    ia = 0
    ib = 0

    num_corr1=0
    num_corr2=0
    t=0
    for r2 in res2:
        r1 = None
        for item in res1:
            if r2.file_name.rpartition('-')[0] == item.file_name.split('.')[0]:
                r1 = item
        if r1 == None: continue
        print(r2)
        num_corr2 += 1 if r2.is_lower_class == lower else 0
        print(r1)
        num_corr1 += 1 if r1.is_lower_class == lower else 0
        t+=1
        print()

    print("-------------")
    print(f"percent correct of {str} b4 isolation: {((num_corr2/t) * 100):.2f}%")
    print(f"percent correct of {str} after isolation: {((num_corr1/t) * 100):.2f}%")

model = DeepfakeClassificationModel(result_handler=DfResultHandler(-3, "Fake", "Real", 10, .95))

print()
print("Authentic files")
print_res(model.evaluate_folder("../AuthenticSoundFiles",progress_bar=True), model.evaluate_folder("../AuthenticSoundFiles/separated",progress_bar=True), "real", False)

print("\n")
print("Deepfake files")
print_res(model.evaluate_folder("../DeepfakeSoundFiles",progress_bar=True), model.evaluate_folder("../DeepfakeSoundFiles/separated",progress_bar=True), "fake", True)
print("\n")




