import tweepy
from tqdm import tqdm

def read_tweet_ids(filename):
    with open(filename, 'r') as tweets:
        next(tweets)
        ids = []
        for line in tweets:
            id = line.strip().split('\t')
            ids.append(id)
        return ids


# Get your Twitter API credentials and enter them here
consumer_key = "d9CMkt6X2eL3q5cVJUPCQrOkW"
consumer_secret = "b1cHBkhlezqiDOODyEdRXLAqgnr8olRHIkIZz4YTnagonNr8vX"
access_key = "3051644665-awIDV3KPqtwHSGxwBtDsze1IBGfOylsK9o2snAq"
access_secret = "DgITG6OJNiGXQyTAeOmewkj9cRHudxKARjZYd5raeFHdS"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


tw = read_tweet_ids('spanglish_corpus.txt')

tweet_count, new_tweets = 0, []

for tweet in tqdm(tw):
    try:
        out = api.get_status(tweet[2]).text
        new_tweets.append(out)
        tweet_count += 1
    except tweepy.error.TweepError:
        pass


print("tweet count:",tweet_count)
with open('%d_more_tweets.txt' % tweet_count, 'w') as tweets_out:
    for t in new_tweets:
        tweets_out.write(str(t) + "\n")








