
"""
Full replication script:
1. Download the replication folder
2. Install all modules specified in Pipfile
3. Set your paths in paths.py
4. Run this script
"""

# ----------------------------------------------------------------------------------------------------------------------
# Import modules and set parameters
# ----------------------------------------------------------------------------------------------------------------------

# Modules
import os
import subprocess
import sys
import glob
from pathlib import Path

# Import paths from file "paths,py" (should be placed in same folder)
from paths import *
print("Input data should be located at:", main_data_path)
print("Output will be saved at:", main_output_path)
print("Code should be located at:", code_path)

# Set parameters
train_models = False  # Set to True if models should be trained (may take days/weeks)

# ----------------------------------------------------------------------------------------------------------------------
# Create folder structure for output directory
# ----------------------------------------------------------------------------------------------------------------------

# Change to path where output is to be saved
os.chdir(main_output_path)

# Create main output folders
main_folders = ["train",
                "results",
                "exhibits"]

for folder in main_folders:
    try:
        os.mkdir(folder)
    except FileExistsError:
        print(f"Directory {folder} already exists.")

# Create subfolders in training data folder
train_folders = ["mx",
                 "mx/weeks",
                 "mx/days",
                 "us",
                 "us/weeks",
                 "us/days"]

for folder in train_folders:
    try:
        os.mkdir(f"train/{folder}")
    except FileExistsError:
        print(f"Directory train/{folder} already exists.")

# Create subfolders in results folder
results_folders = ["mx",
                   "mx/hyperparams",
                   "mx/hyperparams/preds",
                   "mx/stacking_preds",
                   "mx/bias_correction",
                   "mx/validity",
                   "us",
                   "us/hyperparams",
                   "us/hyperparams/preds",
                   "us/stacking_preds",
                   "us/bias_correction",
                   "us/validity"]

for folder in results_folders:
    try:
        os.mkdir(f"results/{folder}")
    except FileExistsError:
        print(f"Directory results/{folder} already exists.")

# Create subfolders in exhibits folder
exhibit_folders = ['figures',
                   r'tables',
                   r'figures/analysis',
                   r'figures/bias',
                   r'figures/shap',
                   r'figures/simulations',
                   r'figures/subsample',
                   r'figures/validity',
                   r'tables/appendix',
                   r'tables/appendix/feature_descriptions',
                   r'tables/appendix/feature_statistics']

for folder in exhibit_folders:
    try:
        os.mkdir(f"exhibits/{folder}")
    except FileExistsError:
        print(f"Directory exhibits/{folder} already exists.")


# ----------------------------------------------------------------------------------------------------------------------
# Run all files
# ----------------------------------------------------------------------------------------------------------------------

# Change to code directory
os.chdir(code_path)

# Define function to run all py files in a folder
def run_all_in_folder(folder):
    folder_scripts = list(folder.glob("*.py"))
    print("Scripts in folder:", folder_scripts)
    for script in folder_scripts:
        print(f"Running {script}...")
        try:
            subprocess.run(
                [sys.executable, str(script)],
                check=True,
                cwd=str(code_path),  # Set working directory
                env={**os.environ, "PYTHONPATH": str(folder.parent), "PYTHONWARNINGS": "ignore"})
            print(f"{script} completed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script}: {e}\n")


# Run data inspection scripts
folder = Path(fr"{code_path}\1_inspect_data")
run_all_in_folder(folder)

# Run model training scripts
# (Note This may take days/weeks. Set shapl=False to speed up by not computing shapley values)
if train_models:
    folder = Path(fr"{code_path}\2_train_models\MX")
    run_all_in_folder(folder)

    folder = Path(fr"{code_path}\2_train_models\US")
    run_all_in_folder(folder)

    folder = Path(fr"{code_path}\3_apply_bias_correction")
    run_all_in_folder(folder)

# Run files to create exhibits (folder: 4_visualize_results"
folder = Path(fr"{code_path}\4_visualize_results")
run_all_in_folder(folder)
