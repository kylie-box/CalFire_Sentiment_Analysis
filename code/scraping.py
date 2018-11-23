import tweepy
import pandas as pd
import re


consumer_key = "Y9AQfYvR6robkHSUWq7SNpg63"
consumer_secret = "FVHydmvOQzl0MP9VHTwgdx6YoR1t4bYGE2JdX1pTM0IYfjVXHY"
access_key = "3494260037-ySvUJ8e7A80CitH64Ec7V6xAzPidPS05e8YTh1n"
access_secret = "AsoMr0k5KPWcyJpCjreZnxkYpmW3ORlhFHcJQFqIF0EF4"


# Function to extract tweets
# def get_tweets(username):
    # Authorization to consumer key and consumer secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Access to user's access key and access secret
auth.set_access_token(access_key, access_secret)

# Calling api
api = tweepy.API(auth,wait_on_rate_limit=True,timeout=100)


date = "2018-11-18"


for i in range(19,23):
    # ct = 0
    c = pd.DataFrame(columns=['tweet',
                              'time',
                              # 'geo',
                              # 'user_name',
                              # 'user_desc',
                              # 'user_loc',
                              # 'verified'
                              ])
    date = '2018-11-{}'.format(i)
    out_date = '2018-11-{}'.format(i-1)

    for tweet in tweepy.Cursor(api.search,q='#WoolseyFire OR #LAFire OR #CampFire OR "camp fire" OR "california fire"-filter:retweets',
                               lang="en",
                               until=date,
                               tweet_mode='extended').items(300):
        youtube_share = r'YouTube'
        if re.search(youtube_share, tweet.full_text, re.IGNORECASE) or len(tweet.full_text.split()) <= 2:
            print(tweet.full_text)
            print("follower count: {}".format(tweet.author.followers_count))
            continue

        # if tweet.author.verified and ct < 10:

        c = c.append({'tweet': tweet.full_text,
                      'time': str(tweet.created_at),
                      # 'geo':tweet.geo,
                      # 'user_name':tweet.author.name,
                      # 'user_desc':tweet.author.description,
                      # 'user_loc':tweet.author.location,
                      # 'verified':tweet.author.verified
                      }, ignore_index=True)
            # ct = ct+1
        # elif not tweet.author.verified:
        #     c = c.append({'tweet': tweet.full_text,
        #                   'time': str(tweet.created_at),
        #                   # 'geo': tweet.geo,
        #                   # 'user_name': tweet.author.name,
        #                   # 'user_desc': tweet.author.description,
        #                   # 'user_loc': tweet.author.location,
        #                   # 'verified': tweet.author.verified
        #                   }, ignore_index=True)
        # else:
        #     pass

    c.to_csv('./fire-{}.csv'.format(out_date), index=False, encoding='utf-8')
