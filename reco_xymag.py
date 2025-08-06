import torch 
import numpy as np 
import os 
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
from ResNet.ResNet import ResNet50
import proton64_dataloader 

def loss_function(pred, truth):
	
	# mse = nn.MSELoss()(pred, truth) #l2 loss 

	l1 = nn.L1Loss()(pred, truth) #l1 loss 

	return l1 

####### Parameters #######

checkpoint_dir = 'checkpoints/ResNet50_edep_v2/'
checkpoint_name = checkpoint_dir + 'ResNet50_epoch98.pt'
batch_size = 128

# Evaluate model or train model 
show_results = True 

# Cosmetics 
rescale = True 
disable_tqdm = False 

# Model training
epochs = 100
lr = 0.1
debug = False 
model_name = "ResNet50"

# Find a specific event from the 2D plots
find_point = False 
xx, xy = -350, -160 
fuzzy = 50 
####### End Parameters #######

## Dataloaders 
training_dataset = proton64_dataloader.edepProtons64Train()
validation_dataset = proton64_dataloader.edepProtons64Validation()
train_loader = DataLoader(training_dataset, batch_size=batch_size) 
val_loader = DataLoader(validation_dataset, batch_size=batch_size)

## Save only save three best checkpoints (lowest validation loss)
best_loss = [np.inf, np.inf, np.inf]
best_ckpt = ["", "", ""] 
os.makedirs(checkpoint_dir, exist_ok=True)

## Initilize the model 
model = ResNet50(num_classes=3, channels=1, norm='batch')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  

point_idx = -1 

# Set scale factor 
if rescale: 
	sf = 500 
else: 
	sf = 1 

## Show the results of the model 
if show_results: 
	print("Plotting Results of Model")
	model.load_state_dict(torch.load(checkpoint_name, weights_only=True)['model_state_dict'])
	model.eval() 

	## Get epoch number from checkpoint name 
	match = re.search(r'epoch(\d+)', checkpoint_name)
	if match: epoch_number = int(match.group(1))
	else: epoch_number = "edep"

	with torch.no_grad():
		
		val_truth = [] 
		val_pred = []
		z_truth = [] 

		for idx, data in enumerate(tqdm(val_loader, disable=disable_tqdm)):
			events = data['image'].to(device)
			momentums = data['momentum'].to(device)
			
			events = events.reshape(-1, 1, 64, 64)

			magnitudes = torch.norm(momentums, dim=1).unsqueeze(1)

			truth = torch.cat((momentums[:, :2], magnitudes), dim=1) 

			val_truth.append(truth) 
			z_truth.append(momentums[:, 2])

			## Run the model 
			pred = model(events)
			val_pred.append(pred)	

			## Find a specific point from the 2D pred vs true plots 
			if find_point and point_idx == -1: 
				truth_moms = truth.cpu().numpy() * 500
				pred_moms = pred.cpu().detach().numpy() * 500 
				for idx, (truth_mom, pred_mom) in enumerate(zip(truth_moms, pred_moms)):
					if (xx - fuzzy < truth_mom[0] < xx + fuzzy and 
						xy - fuzzy < pred_mom[0] < xy + fuzzy):
						point_idx = idx
						temp_true = np.round(truth[point_idx].cpu().numpy()*500, 1)
						temp_reco = np.round(pred[point_idx].cpu().detach().numpy()*500, 1)
						true_str = ', '.join(f"{val:6.1f}" for val in temp_true)
						reco_str = ', '.join(f"{val:6.1f}" for val in temp_reco)
						print(f"True: [{true_str}]")
						print(f"Reco: [{reco_str}]")
						break 
				if point_idx != -1: 
					point_truth = truth[point_idx].cpu().numpy() * 500
					point_reco = pred[point_idx].cpu().detach().numpy() * 500
					point_event = events[point_idx].cpu().numpy()
					point_true_z = momentums[point_idx][2].cpu().numpy() * 500
					print("Min =", np.round(np.min(point_event),2), \
		   				", Mean =", np.round(np.mean(point_event[point_event != 0]), 2), \
						", Max =", np.round(np.max(point_event),2))


	## Plot found point (if applicable)
	if find_point and point_idx != -1: 
		point_z_reco = np.sqrt(np.clip(point_reco[2]**2 - point_reco[0]**2 - point_reco[1]**2, a_min=0, a_max=None))
		point_reco = np.append(point_reco, point_z_reco)
		point_truth = np.append(point_truth, np.abs(point_true_z))
		plt.imshow(point_event.squeeze(), cmap='gray')
		plt.tight_layout()
		plt.savefig("my_point.png")
		print("Saved my_point.png")
		plt.clf() 

	## Convert lists to tensors
	val_truth = torch.cat(val_truth, dim=0).cpu().numpy() * sf 
	val_pred = torch.cat(val_pred, dim=0).cpu().detach().numpy() * sf
	z_truth = torch.cat(z_truth, dim=0).cpu().numpy() * sf

	## Reco Pz 	
	z_pred = np.sqrt(np.clip(val_pred[:, 2]**2 - val_pred[:, 0]**2 - val_pred[:, 1]**2, a_min=0, a_max=None))
	z_truth = np.abs(z_truth)
	z_pred = np.expand_dims(z_pred, axis=1)
	z_truth = np.expand_dims(z_truth, axis=1)
	val_pred = np.concatenate((val_pred, z_pred), axis=1)
	val_truth = np.concatenate((val_truth, z_truth), axis=1)

	## Plot Truth vs Prediction
	labels = ["$P_x$", "$P_y$", "Mag", "$P_z$"]
	fig, axes = plt.subplots(1, 4, figsize=(14, 4))
	axes = axes.ravel()

	## Title according to dataset 
	if idx > 500: 
		fig.suptitle("Training, epoch="+str(epoch_number), fontsize=16)
	else: 
		fig.suptitle("Validation, epoch="+str(epoch_number), fontsize=16)

	## Pred vs Truth plots 
	for i in range(len(axes)): 

		## Define ranges 
		if i in [0,1]: # Px, Py 
			threshold = 0.1 * sf
			rmin = -1 * sf
			rmax = 1 *  sf
		if i == 2: # magnitude |P|
			threshold = 0.05 * sf
			rmin = 0.5 * sf
			rmax = 1 * sf 
		if i == 3: # infered abs(Pz)
			threshold = 0.1 * sf
			rmin = 0
			rmax = 1.2 * sf
 
		p_true = val_truth[:, i]
		p_pred = val_pred[:, i]

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

		if find_point and point_idx != -1: 
			axes[i].scatter(point_truth[i], point_reco[i], color='red', s=80, marker="*")

		# Add colorbar
		# cb = fig.colorbar(mesh, ax=ax, shrink=0.7)
		# cb.set_label("Density")

	plt.tight_layout()
	plt.savefig("model_results.png")
	print("Saved model_results.png")

	## End after plotting results 
	exit() 


## Training Loop
for epoch in range(epochs):
	model.train()
	for batch_idx, data in enumerate(tqdm(train_loader, disable=disable_tqdm)):

		## Load Data 
		events = data['image'].to(device)
		momentums = data['momentum'].to(device)
		events = events.reshape(-1, 1, 64, 64)

		## Create vector px, py, |p| 
		magnitudes = torch.norm(momentums, dim=1).unsqueeze(1)
		truth = torch.cat((momentums[:, :2], magnitudes), dim=1) 

		## Run Model 
		optimizer.zero_grad()
		output = model(events)
		loss = loss_function(output, truth)
		loss.backward()
		optimizer.step() 
	scheduler.step() # Update learning rate after each epoch

	## Validation Loop 
	model.eval()
	with torch.no_grad():
		
		val_loss = 0 
		for val_idx, val_data in enumerate(tqdm(val_loader, disable=disable_tqdm)):

			## Load Data
			val_events = val_data['image'].to(device)
			val_momentums = val_data['momentum'].to(device)
			val_events = val_events.reshape(-1, 1, 64, 64)

			val_magnitudes = torch.norm(val_momentums, dim=1).unsqueeze(1)

			val_truth = torch.cat((val_momentums[:, :2], val_magnitudes), dim=1) 

			## Run Model
			val_output = model(val_events)
			val_loss += loss_function(val_output, val_momentums).item()

		val_loss /= len(val_loader)
		print("val loss:", val_loss, len(val_loader))

		## Save Top Three Checkpoints  
		if not debug:
			checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch+1}.pt')
			if val_loss < best_loss[0]:
				best_loss.append(val_loss)
				best_loss.pop(0) 
				print(f"Saving Checkpoint: {checkpoint_path}")
				torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
				best_ckpt.append(checkpoint_path)
				bad_ckpt = best_ckpt.pop(0)
				if bad_ckpt != "":
					os.remove(bad_ckpt)
			
		## Log Loss 
		if not debug:
			with open(os.path.join(checkpoint_dir, model_name+'_loss.log'), 'a') as f:
				f.write(f'Epoch {epoch+1}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n')
		print(f'Epoch {epoch+1}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')


## Save final checkpoint
if not debug: 
	checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch+1}.pt')
	torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
else: 
	checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch+1}_test.pt')
	torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
	print("Saved:", checkpoint_path)