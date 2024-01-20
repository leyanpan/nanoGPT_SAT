# %%
import os
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="llm-reasoning",
)

# Set the path to the data folder
data_folder = "../data"

# Get the list of all folders inside the data folder
folders = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]

# # Iterate over each folder
for folder in folders:
    # Get the list of all text files inside the folder
    files = [file for file in os.listdir(os.path.join(data_folder, folder)) if file.endswith(".txt")]
    
    # Iterate over each text file
    for file in files:
        # Get the path to the text file
        file_path = os.path.join(data_folder, folder, file)
        print(file_path)
        
        # Add the text file to wandb datasets
        data_set = wandb.Artifact(name=file, type="dataset")
        data_set.add_file(file_path)
        wandb.log_artifact(data_set)




# %%
# Download SAT_test.txt from wandb and print its contents
artifact = wandb.use_artifact('llm-reasoning/SAT_test.txt:v0', type='dataset')
artifact_dir = artifact.download()
print(artifact_dir)

# %%



