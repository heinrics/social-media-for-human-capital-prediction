# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:42:21 2020

This script downloads tweets through the Twitter Streaming API and writes
them into a txt-file. 


"""
#-----------------------------------------------------------------------------
## Import modules
#-----------------------------------------------------------------------------

import os
import time
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#-----------------------------------------------------------------------------
## Set directory
#-----------------------------------------------------------------------------

os.chdir("PATH TO FOLDER")

#-----------------------------------------------------------------------------
## Set up access to Twitter API
#-----------------------------------------------------------------------------

CONSUMER_KEY = "INSERT"
CONSUMER_SECRET = "INSERT"
ACCESS_TOKEN = "INSERT"
ACCESS_TOKEN_SECRET = "INSERT"

auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# -----------------------------------------------------------------------------
## Create class to get tweets through Streaming API
# -----------------------------------------------------------------------------

class StdOutListener(StreamListener):
    """Creates a class inheriting from the StreamListener class; Modifies class
    methods to implement time_limit, saving data and error handling/reconnection
    to stream. Note that Streamer disconnects whenever either on_data or on_error
    returns False
    (see: https://github.com/tweepy/tweepy/blob/master/tweepy/streaming.py#L33) """

    def __init__(self, filename, time_limit=60):
        self.filename = filename
        self.start_time = time.time()
        self.limit = time_limit
        super(StdOutListener, self).__init__()

    def on_data(self, data):

        if (time.time() - self.start_time) >= self.limit:
            print(f"{self.limit} seconds are over.")
            return False

        else:
            try:
                #print(data)
                current_hour_string = time.strftime('-%Y-%d-%b-%H')
                file_name = self.filename + current_hour_string + '.txt'
                with open(file_name, 'a') as tf:
                    tf.write(data)
            except Exception as e:
                print(f"Error on_data: {e}")

        return True

    def on_error(self, status):
        if status == 420:
            """corresponds to rate limit ((should not be necessary
            as the Streaming API is not rate limited))"""
            return False
            print("Rate limit exceeded!")        
        print(status)
        return True

# -----------------------------------------------------------------------------
## Set parameters and make class instance
# -----------------------------------------------------------------------------
la = "la_tweets"
time_limit = 60*60*24*62 # Set time
la_coordinates = [-117.12776, -55.61183, -34.7299934555, 32.72083]  # Bounding box for Latin America
l_la = StdOutListener(la, time_limit=time_limit)  # Create instance from StdOutListener class 

streamer_la = Stream(auth, l_la) # Create instance from Streamer class (takes listener class instance as argument)

# -----------------------------------------------------------------------------
## Get all tweets based on parameters 
# -----------------------------------------------------------------------------

start_time = time.time()
while (time.time() - start_time) <= time_limit:
    try:
        streamer_la.filter(locations=la_coordinates)
    except Exception as e:
        time.sleep(1)
        print("Error in while-loop:", e)