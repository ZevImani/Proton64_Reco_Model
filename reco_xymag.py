import torch 
import numpy as np 
import os 
from torch.utils.data import DataLoader
import proton64_dataloader 
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from ResNet.ResNet import Bottleneck, ResNet, ResNet50
import matplotlib.pyplot as plt
import re

def loss_function(pred, truth):
	
	# mse = nn.MSELoss()(pred, truth) #l2 loss 

	l1 = nn.L1Loss()(pred, truth) #l1 loss 

	return l1 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Parameters 
disable_tqdm = False 
epochs = 100
img_channels = 3
model_name = "ResNet50"
checkpoint_dir = 'checkpoints/ResNet50_edep_v2/'
# lr = 0.005
lr = 0.1
batch_size = 128
use_latents = False 
show_results = True  
testing = False   

rescale = True # rescale momentums back to -1000, 1000

find_point = False 
xx, xy = -350, -160 
fuzzy = 50 

## Dataset 
# if testing: 
# 	print("TESTING MODE")
# 	disable_tqdm = True 
# 	# training_dataset = proton64_dataloader.protons64xTrain(only_one=True)
# 	# validation_dataset = proton64_dataloader.protons64xValidation(only_one=True)
# 	training_dataset = proton64_dataloader.protons64xTrain()
# 	validation_dataset = proton64_dataloader.protons64xValidation()
# else: 
# 	# training_dataset = proton64_dataloader.protons64xTrain()
# 	# validation_dataset = proton64_dataloader.protons64xValidation()
# 	training_dataset = proton64_dataloader.protons64Train()
# 	validation_dataset = proton64_dataloader.protons64Validation()

# if use_latents: 
# 	# training_dataset = proton64_dataloader.latent_protons64Train(only_one=True)
# 	# validation_dataset = proton64_dataloader.latent_protons64Validation(only_one=True)
# 	training_dataset = proton64_dataloader.latent_protons64Train()
# 	validation_dataset = proton64_dataloader.latent_protons64Validation()

training_dataset = proton64_dataloader.edepProtons64Train()
validation_dataset = proton64_dataloader.edepProtons64Validation()


## Dataloaders 
train_loader = DataLoader(training_dataset, batch_size=batch_size) 
val_loader = DataLoader(validation_dataset, batch_size=batch_size)

## Messing with the data (commend out for reco model)
# for idx, data in enumerate(tqdm(val_loader, disable=disable_tqdm)):

# 	if idx == 1: 
# 		data1 = data['image'].cpu().numpy()
# 		mom1 = data['momentum'].cpu().numpy() * 500 

# 		# Create a figure and 2x2 grid of subplots
# 		fig, axes = plt.subplots(4, 4, figsize=(8, 8))

# 		# Flatten the axes array for easy iteration
# 		axes = axes.ravel()

# 		for i in range(len(axes)): 
# 			axes[i].imshow(data1[i], cmap='gray')
# 			axes[i].set_title(f"{mom1[i][0]:.1f}  {mom1[i][1]:.1f}  {mom1[i][2]:.1f}", fontsize=11)
# 			axes[i].axis('off')
# 		plt.tight_layout()
# 		plt.savefig("test.png")
# 		exit() 
# 	continue 

# 	if idx == 0: 
# 		data1 = data['image'].cpu().numpy()
# 		mom1 = data['momentum'].cpu().numpy()
# 	if idx == 1: 
# 		data2 = data['image'].cpu().numpy()
# 		mom2 = data['momentum'].cpu().numpy()
# 		my_data = data1 + data2
		
# 		print(my_data.shape)
# 		my_dataset = {
# 			'image1': data1, 'mom1': mom1, 
# 			'image2': data2, 'mom2': mom2,
# 			'image': my_data}

# 		# np.save("double_data_train.npy", my_dataset)

# 		plt.imshow(my_data[0], cmap='gray', vmin=0, vmax=1)
# 		plt.savefig("test.png")

# 		exit()

## Save only save three best checkpoints (lowest validation loss)
best_loss = [np.inf, np.inf, np.inf]
best_ckpt = ["", "", ""] 
os.makedirs(checkpoint_dir, exist_ok=True)

if use_latents: 
	model = ResNet50(num_classes=3, channels=3, norm='batch')
else: 
	model = ResNet50(num_classes=3, channels=1, norm='batch')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
model.to(device)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)  
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
	checkpoint_name = checkpoint_dir + 'ResNet50_epoch98.pt'
	model.load_state_dict(torch.load(checkpoint_name, weights_only=True)['model_state_dict'])
	model.eval() 

	## Get epoch number from checkpoint naame 
	match = re.search(r'epoch(\d+)', checkpoint_name)
	if match: epoch_number = int(match.group(1))
	else: epoch_number = "edep"

	with torch.no_grad():
		
		val_truth = [] 
		val_pred = []
		z_truth = [] 

		backscatters = [] 

		for idx, data in enumerate(tqdm(val_loader, disable=disable_tqdm)):
			events = data['image'].to(device)
			momentums = data['momentum'].to(device)
			if use_latents: 
				events = events.reshape(-1, 3, 16, 16)
				momentums *= 500 
			else: 
				events = events.reshape(-1, 1, 64, 64)

			magnitudes = torch.norm(momentums, dim=1).unsqueeze(1)

			truth = torch.cat((momentums[:, :2], magnitudes), dim=1) 

			val_truth.append(truth) 
			z_truth.append(momentums[:, 2])

			# Run the model 
			pred = model(events)
			val_pred.append(pred)	

			# break 

			## find specific point 
			if find_point and point_idx == -1: 
				# xx, xy = 200, -400 
				# fuzzy = 10 
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
						# print(f"True: {np.array2string(temp_true, separator=', ')}")
						# print(f"Reco: {np.array2string(temp_reco, separator=', ')}")
						break 
				if point_idx != -1: 
					point_truth = truth[point_idx].cpu().numpy() * 500
					point_reco = pred[point_idx].cpu().detach().numpy() * 500
					point_event = events[point_idx].cpu().numpy()
					point_true_z = momentums[point_idx][2].cpu().numpy() * 500
					print("Min =", np.round(np.min(point_event),2), \
		   				", Mean =", np.round(np.mean(point_event[point_event != 0]), 2), \
						", Max =", np.round(np.max(point_event),2))



	if find_point and point_idx != -1: 
		point_z_reco = np.sqrt(np.clip(point_reco[2]**2 - point_reco[0]**2 - point_reco[1]**2, a_min=0, a_max=None))
		point_reco = np.append(point_reco, point_z_reco)
		point_truth = np.append(point_truth, np.abs(point_true_z))
		# point_reco.append(point_z_reco)
		# point_truth.append(np.abs(point_true_z))

		np.save("bad_event.npy", point_event.squeeze())

		plt.imshow(point_event.squeeze(), cmap='gray')
		plt.tight_layout()
		plt.savefig("test_img.png")
		plt.clf() 

	# Convert lists to tensors
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

	# Plot Truth vs Prediction
	fig, axes = plt.subplots(1, 4, figsize=(14, 4))
	axes = axes.ravel()

	# Define error thresholds
	labels = ["$P_x$", "$P_y$", "Mag", "$P_z$"]

	if idx > 500: 
		fig.suptitle("Training, epoch="+str(epoch_number), fontsize=16)
	else: 
		fig.suptitle("Validation, epoch="+str(epoch_number), fontsize=16)

	## Pred vs Truth plots 
	for i in range(len(axes)): 

		# Define ranges 
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

		# Optionally apply mask to data first 
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
	plt.savefig("test.png")
	print("Saved test.png")

	exit() 

# for batch_idx, data in enumerate(tqdm(train_loader, disable=disable_tqdm)):
# 	events = data['image'].to(device)
# 	momentums = data['momentum'].to(device)
# 	print(torch.min(events), torch.mean(events), torch.max(events))
# exit() 

## Training Loop
for epoch in range(epochs):
	model.train()
	for batch_idx, data in enumerate(tqdm(train_loader, disable=disable_tqdm)):

		## Load Data 
		events = data['image'].to(device)
		momentums = data['momentum'].to(device)
		# print(events.shape)
		if use_latents: 
			events = events.reshape(-1, 3, 16, 16)
			momentums *= 500 
		else: 
			events = events.reshape(-1, 1, 64, 64)

		# print(events.shape)
		# exit() 

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
	
	if testing: 
		print(output[0])
		print(truth[0])
		# print(output.shape)
		# print(truth.shape)
		# exit()
		# show_idx = 3
		# # loss = loss_function(output[0], momentums[0])
		# pred = output[show_idx].detach().cpu().numpy() * 500 
		# # truth = momentums[show_idx].cpu().numpy() * 500 
		# truth = truth[show_idx]
		# # print(np.round(truth, 2),"\t", np.round(pred, 2))#, "\t", np.round(loss.item(),4))
		# x,y,z = output[show_idx].detach().cpu().numpy()
		# x1, y1, z1 = truth[show_idx].cpu().numpy()
		# print(f"({x:.2f}, {y:.2f}, {z:.2f})") 
		# print(f"({x1:.2f}, {y1:.2f}, {z1:.2f})") 


	## Validation Loop 
	model.eval()
	with torch.no_grad():
		
		val_loss = 0 
		for val_idx, val_data in enumerate(tqdm(val_loader, disable=disable_tqdm)):

			## Load Data
			val_events = val_data['image'].to(device)
			val_momentums = val_data['momentum'].to(device)
			if use_latents: 
				val_events = val_events.reshape(-1, 3, 16, 16)
				momentums *= 500
			else: 
				val_events = val_events.reshape(-1, 1, 64, 64)

			val_magnitudes = torch.norm(val_momentums, dim=1).unsqueeze(1)

			val_truth = torch.cat((val_momentums[:, :2], val_magnitudes), dim=1) 


			## Run Model
			val_output = model(val_events)
			val_loss += loss_function(val_output, val_momentums).item()
		val_loss /= len(val_loader)
		print("val loss:", val_loss, len(val_loader))

		## Save Top Three Checkpoints  
		if not testing:
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
		if not testing:
		# if 1: 
			with open(os.path.join(checkpoint_dir, model_name+'_loss.log'), 'a') as f:
				f.write(f'Epoch {epoch+1}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n')
		print(f'Epoch {epoch+1}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')


## Save final checkpoint
if not testing: 
	checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch+1}.pt')
	torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
else: 
	checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch+1}_test.pt')
	torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
	print("Saved:", checkpoint_path)