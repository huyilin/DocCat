# -*- coding: utf-8 -*-
__author__ = 'anna'

# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json
import sys

reload(sys)

sys.setdefaultencoding('utf8')

# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
import ConfigParser

config = ConfigParser.RawConfigParser()
configFilePath = 'token_anna.conf'
config.read(configFilePath)

ACCESS_TOKEN = config.get('token', 'ACCESS_TOKEN')
ACCESS_SECRET = config.get('token', 'ACCESS_SECRET')
CONSUMER_KEY = config.get('token', 'CONSUMER_KEY')
CONSUMER_SECRET = config.get('token', 'CONSUMER_SECRET')

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

twitter_stream = TwitterStream(auth=oauth)

#iterator = twitter_stream.statuses.sample()
iterator = twitter_stream.statuses.filter(locations= "103.516987, 1.243245, 104.993955 , 1.477511", language = "en")


file_size = 2000
tweet_bag = []
for tweet in iterator:
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