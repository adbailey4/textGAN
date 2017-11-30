#!/usr/bin/env python
"""Scrape twitter for data"""
########################################################################
# File: scrape_twitter.py
#  executable: scrape_twitter.py
#
# Author: Andrew Bailey
# History: 11/29/17 Created
########################################################################

import tweepy
from tweepy import OAuthHandler
import numpy as np
import time
import random
import sys
import os
import json


def load_json(path):
    """Load a json file and make sure that path exists"""
    path = os.path.abspath(path)
    assert os.path.isfile(path), "Json file does not exist: {}".format(path)
    with open(path) as json_file:
        args = json.load(json_file)
    return args

# You have to get your own keys to generate this data. This article describes how to access the twitter API
# and some basic tweepy usages.
# https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/


key_data = load_json(sys.argv[1])

consumer_key = key_data["consumer_key"]
consumer_secret = key_data["consumer_secret"]
access_token = key_data["access_token"]
access_secret = key_data["access_secret"]

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


class RandomTwitterData(object):
    """Scrape a bunch of random tweets from different users"""

    def __init__(self, api):
        self.api = api
        self.user_ids = []
        self.tweets = []

    def generate_users(self, pop_size, sample_size, per_user=10):
        """Generate a bunch of random users"""
        ids = []
        try:
            for friend in tweepy.Cursor(api.friends_ids).items(per_user):
                ids.append(friend)
        except tweepy.TweepError as e:
            print(e)
            pass
        # iterate over ids and generate more users
        while len(ids) < pop_size:
            try:
                for friend in tweepy.Cursor(api.friends_ids, id=ids[np.random.randint(0, len(ids))]).items(per_user):
                    ids.append(friend)
            except tweepy.TweepError as e:
                print(e)
                time.sleep(60)
                pass
        self.user_ids = random.sample(ids, sample_size)

    def scrape_random_tweets(self, total, per_user):
        """Get a bunch of random tweets from random users"""
        num_ids = len(self.user_ids)
        while len(self.tweets) < total:
            try:
                for status in tweepy.Cursor(api.user_timeline,
                                            user_id=self.user_ids[np.random.randint(0, num_ids)]).items(per_user):
                    self.tweets.append(status.text)
            except tweepy.TweepError as e:
                print(e)
                time.sleep(60)
                pass


scraper = RandomTwitterData(api)
scraper.generate_users(100, 10, per_user=10)
scraper.scrape_random_tweets(100, 10)
print(len(scraper.tweets))
