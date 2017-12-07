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
import csv
import unicodecsv


def load_json(path):
    """Load a json file and make sure that path exists"""
    path = os.path.abspath(path)
    assert os.path.isfile(path), "Json file does not exist: {}".format(path)
    with open(path) as json_file:
        args = json.load(json_file)
    return args


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
            for friend in tweepy.Cursor(self.api.friends_ids).items(per_user):
                ids.append(friend)
        except tweepy.TweepError as e:
            print("Sleeping for 60 seconds")
            print(e)
            pass
        # iterate over ids and generate more users
        while len(ids) < pop_size:
            try:
                for friend in tweepy.Cursor(self.api.friends_ids, id=ids[np.random.randint(0, len(ids))]).items(per_user):
                    ids.append(friend)
            except tweepy.TweepError as e:
                print(e)
                print("Sleeping for 60 seconds")
                time.sleep(60)
                pass
        self.user_ids = random.sample(ids, sample_size)

    def scrape_random_tweets(self, total, per_user):
        """Get a bunch of random tweets from random users"""
        num_ids = len(self.user_ids)
        tweets= []
        while len(self.tweets) < total:
            try:
                for status in tweepy.Cursor(self.api.user_timeline,
                                            user_id=self.user_ids[np.random.randint(0, num_ids)]).items(per_user):
                    tweets.append(status.text)
            except tweepy.TweepError as e:
                print(e)
                print("Sleeping for 60 seconds")
                time.sleep(60)
                pass
        edited_tweets = self.remove_retweet(tweets)
        self.tweets.append(edited_tweets)
        return edited_tweets

    @staticmethod
    def remove_retweet(tweets):
        """Removes the RT in front of retweets"""
        all_good_tweets = []
        for tweet in tweets:
            if tweet.startswith("RT"):
                all_good_tweets.append(tweet[3:])
            else:
                all_good_tweets.append(tweet)
        return all_good_tweets


class MyStreamListener(tweepy.StreamListener):

    def __init__(self, api):
        self.api = api
        self.output_csv = "/Users/andrewbailey/CLionProjects/nanopore-RNN/textGAN/example_tweet_data/train_csv/random_tweets.csv"
        self.counter = 0
        csv_file = open(self.output_csv, 'w+')
        self.w = unicodecsv.writer(csv_file, encoding='utf-8')
        self.w.writerow(("handle", "tweet"))
        super(MyStreamListener, self).__init__(api)

    def on_status(self, status):
        """Removes the RT in front of retweets"""
        tweet = status.text
        if tweet.startswith("RT"):
            self.w.writerow([self.counter, tweet[3:]])
        else:
            self.w.writerow([self.counter, tweet[3:]])
        self.counter += 1
        if self.counter % 300 == 0:
            print("Saved {} tweets".format(self.counter))

    def on_error(self, status_code):
        if status_code == 420:
            print(status_code)
            #returning False in on_data disconnects the stream
            return False


# You have to get your own keys to generate this data. This article describes how to access the twitter API
# and some basic tweepy usages.
# https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/


def main():
    key_data = load_json(sys.argv[1])

    consumer_key = key_data["consumer_key"]
    consumer_secret = key_data["consumer_secret"]
    access_token = key_data["access_token"]
    access_secret = key_data["access_secret"]

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    keywords = ['twitter']
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
    myStreamListener = MyStreamListener(api)
    myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
    print("Connected to Twitter")
    myStream.filter(track=keywords, languages=['en'], async=False)

    # scraper = RandomTwitterData(api)
    # scraper.generate_users(10, 5, per_user=5)
    # scraper.scrape_random_tweets(10, 10)
    # output_csv = "/Users/andrewbailey/CLionProjects/nanopore-RNN/textGAN/example_tweet_data/train_csv/100_random_tweets.csv"
    # with open(output_csv, 'w+') as csv_file:
    #     w = unicodecsv.writer(csv_file, encoding='utf-8')
    #     w.writerow(("handle", "tweet"))
    #     w.writerows([[i, tweet] for i, tweet in enumerate(scraper.tweets)])


if __name__ == '__main__':
    main()
