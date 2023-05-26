import tweepy
from tweepy import OAuthHandler
import pandas as pd

"""I like to have my python script print a message at the beginning. This helps me confirm whether everything is set up correctly. And it's nice to get an uplifting message ;)."""

print("You got this!")

access_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
consumer_key = 'XXXXXXXXXXXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

tweets = []

count = 1

"""Twitter will automatically sample the last 7 days of data. Depending on how many total tweets there are with the specific hashtag, keyword, handle, or key phrase that you are looking for, you can set the date back further by adding since= as one of the parameters. You can also manually add in the number of tweets you want to get back in the items() section."""

for tweet in tweepy.Cursor(api.search, q="hai", count=450, since='2012-02-28', tweet_mode="extended").items(1000):
	
        if (tweet.lang != 'en' or tweet.truncated == True):
            continue

        print(count)
        count += 1

        try:
            # data = [tweet.created_at, tweet.id, tweet.text, tweet.user._json['screen_name'], tweet.user._json['name'], tweet.user._json['created_at'], tweet.entities['urls']]
            # data = tuple(data)
            # tweets.append(data)
            tweets.append(tweet.full_text + " --- " + str(tweet.truncated))

        except tweepy.TweepError as e:
            print(e.reason)
            continue

        except StopIteration:
            break

# df = pd.DataFrame(tweets, columns = ['created_at','tweet_id', 'tweet_text', 'screen_name', 'name', 'account_creation_date', 'urls'])
df = pd.DataFrame(tweets, columns = ['tweet_text'])

"""Add the path to the folder you want to save the CSV file in as well as what you want the CSV file to be named inside the single quotations"""
df.to_csv(path_or_buf = '/pathLocation', index=False)
