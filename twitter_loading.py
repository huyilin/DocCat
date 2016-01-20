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

i = 0
while True:
    try:
        twitter_stream = TwitterStream(auth=oauth)
        iterator = twitter_stream.statuses.filter(locations= "103.516987, 1.243245, 104.993955 , 1.477511", language = "en")
        for tweet in iterator:
            if 'text' in tweet:
                content = tweet['created_at'] + ',' +tweet['text'] + '\n'
            file = open('data/' + '2015-10-11' + '.csv', 'a')
            file.write(content)
            i = i + 1
            print i
            file.close()
            tweet_bag = []
    except:
        pass

