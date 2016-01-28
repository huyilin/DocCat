# -*- coding: utf-8 -*-
__author__ = 'anna'
from datetime import datetime

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
        # iterator = twitter_stream.statuses.filter(locations= "103.588489, 1.233231, 103.994018 , 1.487514", language = "en")
        iterator = twitter_stream.statuses.filter(language = "en")
        for tweet in iterator:
            if 'text' in tweet:
                content = tweet['created_at'] + ' || ' +tweet['text'] + ' || '' + \n'
                print content
            date = str(datetime.utcnow().now().date())
            file = open('download/' + date + '.csv', 'a')
            file.write(content)
            i = i + 1
            print i
            file.close()
            tweet_bag = []
    except:
        pass

