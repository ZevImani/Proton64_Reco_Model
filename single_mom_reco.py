import torch 
import numpy as np 
import os 
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from ResNet.ResNet import ResNet50
import proton64_dataloader 

### Parameters ###
batch_size = 128
rescale = True 
checkpoint_name = 'checkpoints/ResNet50_edep_v2/ResNet50_epoch98.pt'
data_dir = "/n/home11/zimani/latent-diffusion/one_mom_sample/edep_ldm_sample1/"

# sample_mom = [199.1, 307.1, 382.2] # bad_reco sample
sample_mom = [314.0, -126.4, 249.1] # sample1 

save_diff = True 
save_diff_percent = True 
save_2d_hists = True 

##################

## Load the model from checkpoint in eval mode 
model = ResNet50(num_classes=3, channels=1, norm='batch')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load(checkpoint_name, weights_only=True)['model_state_dict'])
model.eval() 

data_range = len(glob.glob(data_dir+"*.npy"))

with torch.no_grad():
	
	val_pred = []

	for batch_num in tqdm(range(data_range)): 

		batch = np.load(data_dir+"batch_"+str(batch_num)+".npy")

		events = torch.tensor(batch).to(device).reshape(-1, 1, 64, 64)

		pred = model(events)

		val_pred.append(pred)  
		print(pred * 500)
		exit()
			
if rescale: 
	sf = 500
else: 
	sf = 1

# Convert lists to tensors
val_pred = torch.cat(val_pred, dim=0).cpu().detach().numpy() * sf 

# plot labels & title 
labels = ["$P_x$", "$P_y$", "Mag", "$P_z$"]
# fig.suptitle("Cond LDM Momentum Variance", fontsize=16)

## Reco Pz 	
z_pred = np.sqrt(np.clip(val_pred[:, 2]**2 - val_pred[:, 0]**2 - val_pred[:, 1]**2, a_min=0, a_max=None))
z_pred = np.expand_dims(z_pred, axis=1)
val_pred = np.concatenate((val_pred, z_pred), axis=1)

true_mom = np.array(sample_mom) / (500 / sf) 

# True px, py, mag, |pz|
true_mag = np.sqrt(true_mom[0]**2 + true_mom[1]**2 + true_mom[2]**2)
true_4mom = np.array([true_mom[0], true_mom[1], true_mag, np.abs(true_mom[2])])
val_truth = np.tile(true_4mom, (val_pred.shape[0], 1))

# Absolute difference 
if save_diff: 
	fig, axes = plt.subplots(1, 4, figsize=(16, 5))
	axes = axes.ravel()
	diff = val_pred - val_truth
	n_events = diff.shape[0]
	for i in range(len(axes)): 

		weights = [1.0 / n_events] * n_events  # Each event contributes equally to sum to 1
		axes[i].hist(diff[:, i], bins=50, histtype='step', weights=weights)
		# axes[i].set_title(f"{labels[i]}")

		if i in [0,1,2]: 
			axes[i].set_xlim(-0.5*sf, 0.5*sf)
		if i == 3: 
			axes[i].set_xlim(-0.5*sf, 0.5*sf)

		axes[i].set_xlabel(f"pred - truth")
		if i == 0: 
			axes[i].set_ylabel(f"Fraction of events")

		# Add vertical lines at ±0.1
		axes[i].axvline(-0.1*sf, color='red', linestyle='--', linewidth=1)
		axes[i].axvline(0.1*sf, color='red', linestyle='--', linewidth=1)

		# Calculate percentage of events within [-0.1, 0.1]
		within_range = ((diff[:, i] >= -0.1*sf) & (diff[:, i] <= 0.1*sf)).sum()
		percent_within = 100 * within_range / n_events

		# Set title with percentage
		axes[i].set_title(f"{labels[i]} ({percent_within:.1f}% in ±50)") 

	plt.tight_layout()
	plt.savefig("single_mom_diffs.png")
	print("Saved single_mom_diffs.png")
	plt.clf()

# Percent difference (from Claud)
if save_diff_percent: 

	fig, axes = plt.subplots(1, 4, figsize=(16, 5))
	axes = axes.ravel()


	# Calculate percent difference instead of absolute difference
	percent_diff = 100 * (val_pred - val_truth) / val_truth
	n_events = percent_diff.shape[0]

	for i in range(len(axes)): 
		weights = [1.0 / n_events] * n_events  # Each event contributes equally to sum to 1
		axes[i].hist(percent_diff[:, i], bins=50, histtype='step', weights=weights)
		
		if i in [0,1,2]: 
			axes[i].set_xlim(-50, 50)  # Adjust xlim for percent values
		if i == 3: 
			axes[i].set_xlim(-50, 50)  # Adjust xlim for percent values
		
		axes[i].set_xlabel(f"(pred - truth) / truth × 100%")
		if i == 0: 
			axes[i].set_ylabel(f"Fraction of events")
		
		# Add vertical lines at ±10% instead of ±0.1*sf
		axes[i].axvline(-10, color='red', linestyle='--', linewidth=1)
		axes[i].axvline(10, color='red', linestyle='--', linewidth=1)
		
		# Calculate percentage of events within [-10%, 10%]
		within_range = ((percent_diff[:, i] >= -10) & (percent_diff[:, i] <= 10)).sum()
		percent_within = 100 * within_range / n_events
		
		# Set title with percentage
		axes[i].set_title(f"{labels[i]} ({percent_within:.1f}% in ±10%)") 
	plt.tight_layout()
	plt.savefig("single_mom_diffs_percent.png")
	print("Saved single_mom_diffs_percent.png")
	plt.clf()

if save_2d_hists:

	fig, axes = plt.subplots(1, 4, figsize=(16, 5))
	axes = axes.ravel()

	for i in range(len(axes)): 

		# Define ranges 
		if i in [0,1]: # Px, Py 
			threshold = 0.1 * sf
			rmin = -2 * sf 
			rmax = 2 * sf
		if i == 2: # magnitude |P|
			threshold = 0.05 * sf
			rmin = 0.6 * sf
			rmax = 2 * sf
		if i == 3: # infered abs(Pz)
			threshold = 0.1 * sf
			rmin = 0 * sf
			rmax = 2 * sf

		# Optionally apply mask to data first 
		p_true = val_truth[:, i]#[mismatch_x_mask]
		p_pred = val_pred[:, i]#[mismatch_x_mask]

		# Compute accuracy within threshold
		within_threshold = np.abs(p_true - p_pred) <= threshold
		accuracy = np.mean(within_threshold) * 100  # Convert to percentage
		
		# Define bin size
		bin_size = 100
		x_edges = np.linspace(rmin, rmax, bin_size)
		y_edges = x_edges

		# Compute 2D histogram
		hist, x_edges, y_edges = np.histogram2d(p_true, p_pred, bins=(x_edges, y_edges))
		
		# Plot the density heatmap
		mesh = axes[i].pcolormesh(x_edges, y_edges, hist.T, cmap='viridis', shading='auto', norm=plt.cm.colors.LogNorm())

		axes[i].set_xlim(rmin, rmax)
		axes[i].set_ylim(rmin, rmax)
		axes[i].set_aspect('equal')

		## y=x with error bars
		x = np.linspace(min(x_edges), max(x_edges), 100)
		axes[i].plot(x, x, 'red', alpha=0.5, linestyle='-', label="x = y")
		axes[i].plot(x, x + threshold, 'red', linestyle='dashed', alpha=0.5, label=f"+{threshold:.3f}")
		axes[i].plot(x, x - threshold, 'red', linestyle='dashed', alpha=0.5, label=f"-{threshold:.3f}")
		
		axes[i].set_xlabel(f"True")# {labels[i]}")
		if i == 0: 
			axes[i].set_ylabel(f"Predicted {labels[i]}")

		axes[i].set_title(f"{labels[i]} accuracy ±{threshold} = {accuracy:.2f}% ")
		# axes[i].legend()

		# Add colorbar
		# cb = fig.colorbar(mesh, ax=ax, shrink=0.7)
		# cb.set_label("Density")

	plt.tight_layout()
	plt.savefig("single_mom_2d_hists.png")
	print("Saved single_mom_2d_hists.png")
	plt.clf()
