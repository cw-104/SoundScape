import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for (model_features, audio_features), labels in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(model_features, audio_features)

            # Calculate loss
            loss = criterion(outputs, targets.long())
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    # Print epoch loss
    avg_loss = running_loss / len(data_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


def evaluate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():
        for model_features, audio_features, labels in dataloader:
            outputs = model(model_features, audio_features)
            predicted = (outputs.squeeze() > 0.5).float()  # Convert probabilities to binary predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')



