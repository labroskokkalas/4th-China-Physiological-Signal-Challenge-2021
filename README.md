# Python code for the 4th China Physiological Signal Challenge 2021

## What's in this repository?

We implemented a Convolutional recurrent neural network that uses the time representations of the ECG lead signals as features.

The code uses two main scripts, as described below, to train and run the model for the 2021 Challenge.

## How do I install the requirements?

run
    
	pip install -r requirements.txt

## How do I train the model?

model is trained on Training Set II

run 

    python train_model.py <data_path> <run_path>
	
where <data_path> is the folder path of the training set II, <run_path> is the folder of the generated training data and the generated model. Training mode needs a GPU to run faster. 	

## How do I run the model?

run 

    python entry_2021.py <data_path> <result_save_path>

where <data_path> is the folder path of the test set, <result_save_path> is the folder path of the exported signals data and the detection results. Inference mode does not need  GPU 



