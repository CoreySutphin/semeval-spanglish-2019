import tweepy


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


tw = read_tweet_ids('spanglish_corpus.txt')[:100]

tweet_count = 0
for tweet in tw:
    try:
        out = api.get_status(tweet[2])
        print(out.text.encode("utf-8"))
        tweet_count += 1
    except:
        pass

print(tweet_count)





