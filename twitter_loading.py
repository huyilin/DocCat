__author__ = 'anna'
# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

# Variables that contains the user credentials to access Twitter API
ACCESS_TOKEN = '3461539752-zDImdWiXLQTVHFiKIKG556lMDValtA8jyhDbr7a'
ACCESS_SECRET = 'JMLJRb5f3gEbfqKIxYh2Rmbl4UZWQIjUPxdNjQnpuHIcg'
CONSUMER_KEY = '6GdsrUCKYF0abXvGSG2b4neoX'
CONSUMER_SECRET = 'j9wHCm0a2NA7N4iEwd5W1QK344RB1VGucVfHOi6eb2bMBUjzIP'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

# Initiate the connection to Twitter Streaming API
twitter_stream = TwitterStream(auth=oauth)

# Get a sample of the public data following through Twitter
iterator = twitter_stream.statuses.sample()

# Print each tweet in the stream to the screenÅ
# Here we set it to stop after getting 1000 tweets.
# You don't have to set it to stop, but can continue running
# the Twitter API to collect data for days or even longer.
# this is the case
tweet_count = 500
for tweet in iterator:
    tweet_count -= 1
    # Twitter Python Tool wraps the data returned by Twitter
    # as a TwitterDictResponse object.
    # We convert it back to the JSON format to print/score
    if 'text' in tweet:
        print tweet['text']
    # The command below will do pretty printing for JSON data, try it out
    if tweet_count <= 0:
        break
#huyilinshigou
#I love anna
#I love yilinjkijkj