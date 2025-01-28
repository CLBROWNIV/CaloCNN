import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Function to read the event file and convert into 8x8 tensors
def read_event_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    events = []
    current_event = []

    for line in lines:
        if line.strip():  # Non-empty line
            current_event.append(list(map(int, line.split())))
            if len(current_event) == 8:  # 8 rows complete one event
                event = torch.tensor(current_event, dtype=torch.float32)  # Convert to tensor
                pedestal = event.mean()  # Compute the pedestal as the mean value
                event = torch.clamp(event - pedestal, min=0)  # Subtract the pedestal and clamp at 0
                events.append(event)  # Append the cleaned event
                current_event = []  # Reset for the next event

    return torch.stack(events)  # Combine all events into a single tensor

# Function to compute the energy for each event
def parse_and_sum(file_path):
    sums = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Non-empty line
                # Convert line into a list of numbers
                numbers = [max(0, float(num)) for num in line.split()]
                # Sum the positive numbers
                line_sum = sum(numbers)
            else:
                # Assign zero if the line is empty
                line_sum = 0.0
            # Store the sum
            sums.append(line_sum)

    
    # Convert the list of sums into a PyTorch tensor
    tensor = torch.tensor(sums)
    return tensor

# Custom Dataset for PyTorch
class EnergyDataset(Dataset):
    def __init__(self, events_file, energy_file):
        self.events = read_event_file(events_file)
        self.energies = parse_and_sum(energy_file)

    def __len__(self):
        return len(self.energies)

    def __getitem__(self, idx):
        return self.events[idx].unsqueeze(0), self.energies[idx]  # Add channel dimension for CNN input

# Define the Neural Network
class EnergyPredictor(nn.Module):
    def __init__(self):
        super(EnergyPredictor, self).__init__()
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # (8x8x1 -> 8x8x16)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # (8x8x16 -> 8x8x32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # (8x8x32 -> 8x8x64)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)                                     # (8x8x64 -> 4x4x64)

        # Fully connected layers for energy prediction
        self.fc1 = nn.Linear(64 * 4 * 4, 256)                             # Flatten -> Dense
        self.fc2 = nn.Linear(256, 1)                                      # Energy regression

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # Downsample to 4x4
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Final energy output
        return x

# Training Function
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=10):
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()  # Adjust the learning rate
        loss_history.append(running_loss / len(dataloader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}")

    return loss_history

# Main logic
if __name__ == "__main__":
    # File paths
    events_file = "output.txt"  # Replace with your events file
    energy_file = "energy.txt"
    # Step 1: Create the Dataset and DataLoader
    dataset = EnergyDataset(events_file, energy_file)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # Step 2: Initialize the Model
    model = EnergyPredictor()

    # Step 3: Define Loss Function and Optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=.1)

    # Step 4: Train the Model
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    model.to(device)
    loss_history = train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=3000)

    np.savetxt("TEST2.txt", loss_history, delimiter=",", fmt="%.6f")  # Save loss as float

    plt.plot(range(1, len(loss_history) + 1), loss_history, color='k', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.yscale('log')
    plt.savefig("TEST2.png")

    # Step 5: Save the Model
    torch.save(model.state_dict(), "TEST2.pth")
    print("Model training complete and saved!")
