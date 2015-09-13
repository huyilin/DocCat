# -*- coding: utf-8 -*-
__author__ = 'anna'

# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json
import sys
reload (sys)
sys.setdefaultencoding('utf8')

# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
import csv

# Variables that contains the user credentials to access Twitter API
ACCESS_TOKEN = '3461539752-zDImdWiXLQTVHFiKIKG556lMDValtA8jyhDbr7a'
ACCESS_SECRET = 'JMLJRb5f3gEbfqKIxYh2Rmbl4UZWQIjUPxdNjQnpuHIcg'
CONSUMER_KEY = '6GdsrUCKYF0abXvGSG2b4neoX'
CONSUMER_SECRET = 'j9wHCm0a2NA7N4iEwd5W1QK344RB1VGucVfHOi6eb2bMBUjzIP'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

# Initiate the connection to Twitter Streaming API
twitter_stream = TwitterStream(auth=oauth)

# Get a sample of the public data following through Twitter
# iterator = twitter_stream.statuses.sample()
iterator = twitter_stream.statuses.filter(locations= "103.516987, 1.243245, 104.993955 , 1.477511", language = "en")
# Print each tweet in the stream to the screen
# Here we set it to stop after getting 1000 tweets.
# You don't have to set it to stop, but can continue running
# the Twitter API to collect data for days or even longer.
# this is the case

file_size = 2000
tweet_bag = []
for tweet in iterator:
    # Twitter Python Tool wraps the data returned by Twitter
    # as a TwitterDictResponse object.
    # We convert it back to the JSON format to print/score
    if 'text' in tweet:
        content = tweet['text'] + '\n'
        tweet_bag.append(content)
    # The command below will do pretty printing for JSON data, try it out
    print len(tweet_bag)
    if len(tweet_bag) == file_size:
        file = open('data/' + str(id(tweet_bag)) + '.csv', 'w')
        for entry in tweet_bag:
            file.write(entry)
        file.close()
        tweet_bag = []