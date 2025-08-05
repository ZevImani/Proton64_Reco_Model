import matplotlib.pyplot as plt
import re

file_path = "checkpoints/ResNet50_edep_v2/ResNet50_loss.log"  

epochs = []
training_losses = []
validation_losses = []

with open(file_path, 'r') as file:
	for line in file:
		match = re.search(r"Epoch (\d+), Training Loss: ([\d\.]+), Validation Loss: ([\d\.]+), Learning Rate: ([\d\.]+)", line)
		if match:
			epochs.append(int(match.group(1)))
			training_losses.append(float(match.group(2)))
			validation_losses.append(float(match.group(3)))

plt.figure(figsize=(10, 4))
plt.plot(epochs, training_losses, label='Training Loss', marker='o')
plt.plot(epochs, validation_losses, label='Validation Loss', marker='s')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('ResNet50 Edep Px, Py, |P| Loss')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("loss_plot.png")
print("Saved: loss_plot.png")


