import time
import os
import glob
from datetime import datetime, timedelta
import gzip
import shutil

storage_directory = r'PATH TO FOLDER'

file_list = glob.glob(storage_directory + "/*.txt")
print(file_list[0:5])


for file in file_list:
    try:
            # Zip the file
            with open(file, 'rb') as f_in:
                with gzip.open(file + '.gz', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove the original file
            if os.path.isfile(file + '.gz'):
                os.remove(file)

    except Exception as e:
        print(f"Exception: {e}")
        continue
