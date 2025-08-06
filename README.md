# Proton64_Reco_Model

This code is used to to reconstruct the momentum of proton events (64x64 images). The model utilizes a slighly modified ResNet50 that has been trained on the proton64 dataset.

Note: this code is currently optimized for Zev's VS Code workflow.

## Running the Reco Model 

The script `reco_xymag.py` is used to train or evaluate the model. The functionality is controlled by variables defined within the script defined at the top in the parameters section. 

There are three main uses: 

1. Training the model. 
2. Evaluating the model on the validation dataset. 
3. Finding outlier events in the validation dataset. 

## Single Momentum Sample

The script `single_mom_reco.py` is used to analyze a sample of proton events generated using a single momentum. Again the functionality is controlled by variables defined at the start of the script. 

`checkpoint_name` defines the model to run. 

`data_dir` points to the directory containing the sample events. It expects files named `batch_X.npy` where X is sequential numbering from zero. 

`sample_mom` is the momentum that all the samples are generated from. Formatted as a python array [px, py, pz]

The desired output visualizations can be specified by the three `save_` parameters. 

