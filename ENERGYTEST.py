import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class EnergyPredictor(nn.Module):
    def __init__(self):
        super(EnergyPredictor, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # Downsample to 4x4
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Final energy output
        return x


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

    return torch.stack(events).unsqueeze(1)  # Combine all events into a single tensor

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
    tensor = torch.tensor(sums, dtype=torch.float32)
    return tensor

def compute_energy(events):
    energies = []
    for event in events:
        max_value = event.max()
        max_indices = (event == max_value).nonzero(as_tuple=True)
        max_row, max_col = max_indices[0][0].item(), max_indices[1][0].item()

        # Define the region of interest (3x3 area around the max value)
        row_start = max(0, max_row - 1)
        row_end = min(8, max_row + 2)
        col_start = max(0, max_col - 1)
        col_end = min(8, max_col + 2)

        # Sum the values in the 3x3 area
        energy = event[row_start:row_end, col_start:col_end].sum().item()
        energies.append(energy)

    return torch.tensor(energies, dtype=torch.float32)

if __name__ == "__main__":
    events_file = "/home/chaslbiv/calorimeterCode/output.txt"
    energy_file = "/home/chaslbiv/calorimeterCode/energy.txt"
    model_path = "/home/chaslbiv/calorimeterCode/TEST2.pth"

    print("Loading test events...")
    events = read_event_file(events_file)
    print(f"Loaded {events.size(0)} test events.")

    print("Loading energies...")
    energy_real = parse_and_sum(energy_file)
    print(f"Loaded {energy_real.size(0)} energies")

    print("Loading 9 bar sums...")
    energy_9_bar = compute_energy(events.squeeze(1))
    print("Loaded")

    print("Loading trained model...")
    model = EnergyPredictor()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.eval()

    print("Predicting energies")
    with torch.no_grad():
        predictions = torch.tensor(model(events).squeeze().numpy(), dtype=torch.float32)

    output_file = "ENERGYTEST_STATS"

    array_old = energy_real.numpy()
    array_new = energy_9_bar.numpy()

    #stats
    energy_9_bar_difference =  torch.abs(energy_9_bar-energy_real)
    energy_prediction_difference = torch.abs(predictions-energy_real)

    energy_9_bar_error = torch.abs((energy_9_bar-energy_real)/energy_real)
    enenrgy_prediction_error = torch.abs((predictions-energy_real)/energy_real)

    e9d = energy_9_bar_difference.numpy()
    epd = energy_prediction_difference.numpy()
    e9e = energy_9_bar_error.numpy()
    epe = enenrgy_prediction_error.numpy()

    dataset = {
        "9 bar sum difference": e9d,
        "Model prediction difference": epd
        #"9 bar sum error": e9e,
        #"Model Prediction error": epe
    }

    for name, data in dataset.items():
        plt.figure(figsize=(8,6))
        plt.hist(data, bins=100, color="black", alpha=0.7, edgecolor='black')
        plt.xlabel("Energy (Error)")
        plt.ylabel("Frequency")
        plt.title(name)

        filename = f"{name}_histogram.png"
        plt.savefig(filename)
        print(f"Saved: {name}")
        plt.close()

    stacked_array = np.column_stack((array_old, array_new))
    cleaned_array = np.nan_to_num(stacked_array, nan=0).astype(float)

    np.savetxt("ENERGYTESTER_RESULTS.txt", cleaned_array, fmt="%d", delimiter="\t")
    print("Saved model predictions")
    print("Program Completed")