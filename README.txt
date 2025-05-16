Replication Package

Steps for Replication

1. Download the Replication Folder
   Ensure all files and subfolders remain in their respective locations.

2. Install Required Modules
   Use the Pipfile to install dependencies:
       pip install pipenv
       pipenv install

3. Set Paths
   Update paths in paths.py:
       - main_data_path: Input data folder
       - main_output_path: Output directory
       - code_path: Code folder

4. Run the Main Script
   Execute the replication script:
       python run_all.py

Folder Structure

Input Data
Place input data in the folder specified by main_data_path.

Output Folders
The script creates these folders automatically:
       - train/: Training data
       - results/: Model outputs
       - exhibits/: Figures and tables

Notes

- Set train_models=True in main_script.py to train models (this can be time-intensive).
- Output is saved in main_output_path.
- The folder \0_prepare_data contains the code to aggregate individual-level to county/municipality-level tweet data. This cannot be run because individual-level data cannot be provided.
