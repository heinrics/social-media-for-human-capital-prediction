Replication Package for "Measuring Human Capital with Social Media Data and Machine Learning"

https://ideas.repec.org/p/bss/wpaper/46.html

In response to persistent gaps in the availability of survey data, a new strand of research leverages alternative data sources through machine learning to track global development. While previous applications have been successful at predicting outcomes such as wealth, poverty or population density, we show that educational outcomes can be accurately estimated using geo-coded Twitter data and machine learning. Based on various input features, including user and tweet characteristics, topics, spelling mistakes, and network indicators, we can account for ~70 percent of the variation in educational attainment in Mexican municipalities and US counties.



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
