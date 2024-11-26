import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

from MapWav2VecSVDDDataset import load_from_disk

print("Creating datasets")

# Initialize feature extractor for Wav2Vec2
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

from datasets import Dataset
import torchaudio

# Load the preprocessed dataset from disk (MapWav2VecSVDDDataset.py to generate)
print("Loading train and dev datasets from disk")
train_dataset, dev_dataset = load_from_disk()

# Load evaluation metric
accuracy = evaluate.load("accuracy")

# Compute metrics function
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# Label mapping
label2id = {0: "bonafide", 1: "deepfake"}
id2label = {v: k for k, v in label2id.items()}

print("\n\n\nInitializing model\n")
# Initialize model
num_labels = len(label2id)
# Use MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
).to(device)

print(f"\nUsing device: {model.device}\n\n")

print("Training model")
# Define training arguments
training_args = TrainingArguments(
    output_dir="sounscape_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    gradient_checkpointing=True
)


# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,  
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
