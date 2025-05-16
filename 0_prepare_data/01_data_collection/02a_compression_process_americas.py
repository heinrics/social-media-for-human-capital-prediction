import time
import os
import glob
from datetime import datetime, timedelta
import gzip
import shutil

storage_directory = 'PATH TO FOLDER'

while True:

    try:
        # Get all files
        file_list = glob.glob(storage_directory + "/*.txt")

        # Remove files newer than two hours ago
        # current hour
        current_hour_string = time.strftime('-%Y-%d-%b-%H')
        # last hour
        last_hour_string = datetime.today() - timedelta(hours=1)
        last_hour_string = last_hour_string.strftime('-%Y-%d-%b-%H')

        print('Start compressing files older than: ', last_hour_string)

        # Iterate over files and compress every file
        for file in file_list:

            try:
                # Leave out current and last hour
                if not (current_hour_string in file or last_hour_string in file):

                    # Zip the file (from stackoverflow:
                    # https: // stackoverflow.com / questions / 8156707 / gzip - a - file - in -python

                    with open(file, 'rb') as f_in:
                        with gzip.open(file + '.gz', 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    # Remove the file
                    if os.path.isfile(file + '.gz'):
                        os.remove(file)

            except Exception as e:
                print(f"Inner exception: {e}")
                continue

        # Sleep a bit more than one hour
        time.sleep(60 * 61)

    except Exception as e:
        print(f"Outer exception: {e}")
        continue
