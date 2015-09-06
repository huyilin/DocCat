__author__ = 'anna'
# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

# Variables that contains the user credentials to access Twitter API
ACCESS_TOKEN = '3563975957-2VQfMjFpzBWFtkkHxTeMvSAPzE6vlVheyv04Qrv'
ACCESS_SECRET = 'vnuncQ46DeZ5fDkmAhx9pIvYuvRWJk7vangvy4IZOcmAy'
CONSUMER_KEY = 'c1F6xszzi0wUSiU9RiMiQY1Se'
CONSUMER_SECRET = 'VVEo1z2N1upbu1LCQXSLVO2f09gVe9VJCSHirK8ZQyZAkE4Ote'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

# Initiate the connection to Twitter Streaming API
twitter_stream = TwitterStream(auth=oauth)

# Get a sample of the public data following through Twitter
iterator = twitter_stream.statuses.sample()

# Print each tweet in the stream to the screen
# Here we set it to stop after getting 1000 tweets.
# You don't have to set it to stop, but can continue running
# the Twitter API to collect data for days or even longer.
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
