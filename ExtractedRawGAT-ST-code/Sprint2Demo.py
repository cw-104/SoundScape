import yaml
import torch
from model import RawGAT_ST 

from evalfile import evaluate_single_file
from evalfolder import evaluate_folder

# Load the model configuration
model_path = "Model/RawGAT_ST_mul/Best_epoch.pth"
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load the model
# Load YAML configuration
with open("model_config_RawGAT_ST.yaml", 'r') as f_yaml:
    config = yaml.safe_load(f_yaml)  

# Extract only the model-related part of the configuration
model_config = config['model']

# Instantiate the model with the correct configuration
model = RawGAT_ST(model_config, device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
print(f"Loaded model from {model_path}")


# evaluation
# file_to_evaluate = "/Users/christiankilduff/Deepfake_Detection_Resources/Datasets/ASVspoof/Eval/ASVspoof2021_DF_eval001/flac/DF_E_2000117.flac"  # Set the path to your FLAC file here
# evaluate_single_file(file_to_evaluate, model, device)

# print("Dataset Sample Data: \n")

# print("Evaluating Sample Deepfake files: \n")
# fake = "sample_data/generated"
# sample_res_fake = evaluate_folder(fake, model, device)


# print("\n\nEvaluating Sample Authentic files: \n")

# real = "sample_data/Real"
# sample_res_real = evaluate_folder(real, model, device)


print("-------------------------")
print("Our Collected Samples:")
print("Evaluating Our Deepfake files: \n")

# fakes = "../DeepfakeSoundFiles/"
fakes = "../../deepfakes"
our_res_fake = evaluate_folder(fakes, model, device)

print("\n\nEvaluating Our Authentic files: \n")

real = "../AuthenticSoundFiles/"
our_res_real = evaluate_folder(real, model, device)

# await keyboard enter


# calc success rate
def success_rate(results, threshold, isReal=True):
    if isReal:
        return sum(1 for _, sc in results if sc >= threshold) / len(results)
    else:
        return sum(1 for _, sc in results if sc < threshold) / len(results)

from tabulate import tabulate

print("Correctly Classified Results")
print("---")
print("Results - DF:\n")
n_df = sum(1 for _, sc in our_res_fake if sc < 0)
print(f"Percentage of Deepfakes: {n_df/len(our_res_fake) * 100:.2f}%")

print()
print("Results - Real")
n_real = sum(1 for _, sc in our_res_real if sc >= 0)
print(f"Percentage of Real: {n_real/len(our_res_real) * 100:.2f}%")
# # Prepare data for the table
# headers = [
#     "Certainty\nThreshold",
#     "Sample\nDeepfake",
#     "Sample Real DF\nFalse Postive"
#     "Collected\nDeepfake",
#     "Collected Real DF\nFalse Postive",
# ]

# data = []
# for i in range(0, 3):
#     row = [
#         f"{-i}",
#         f"{success_rate(sample_res_fake, -i, isReal=False) * 100:.2f}%",
#         f"{(1-(success_rate(sample_res_real, -i,isReal=True)))* 100:.2f}%",
#         f"{success_rate(our_res_fake, -i, isReal=False) * 100:.2f}%",
#         f"{(1-(success_rate(our_res_real, -i, isReal=True))) * 100:.2f}%",
#     ]
#     data.append(row)

# Output the table
# print("\t\t\t\tPercent Correctly Classified %")
# print(tabulate(data, headers=headers, tablefmt="grid"))

# print("\nAuthentic songs that were correctly classified at a threshold of 0:\n")
# authentic = [(_,sc) for _, sc in our_res_real if sc >= 0]
# # print
# for file, score in authentic:
#     print(f"{file} => {score}")

# print("\nAuthentic songs that were correctly classified at a threshold of -1:\n")
# authentic = [(_,sc) for _, sc in our_res_real if sc >= -1 and sc < 0]
# # print
# for file, score in authentic:
#     print(f"{file} => {score}")
