import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize Twitter API
def initialize_twitter_api():
    auth = tweepy.OAuthHandler('CONSUMER_KEY', 'CONSUMER_SECRET')
    auth.set_access_token('ACCESS_TOKEN', 'ACCESS_TOKEN_SECRET')
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

# Collect tweets and perform sentiment analysis
def fetch_and_analyze_tweets(api, stock_name, days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    tweets_data = []
    
    # Fetch tweets for the specified stock keyword
    for tweet in tweepy.Cursor(api.search, q=stock_name, lang="en", since=start_date.date(), until=end_date.date()).items(500):
        tweets_data.append(tweet.text)

    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    daily_sentiments = {}

    # Analyze sentiment for each tweet
    for tweet in tweets_data:
        sentiment = analyzer.polarity_scores(tweet)['compound']
        tweet_date = datetime.now().date()  # Use tweet date if tweet.created_at is enabled
        daily_sentiments.setdefault(tweet_date, []).append(sentiment)

    # Calculate average sentiment score per day
    daily_sentiments_avg = {date: np.mean(scores) for date, scores in daily_sentiments.items()}
    return daily_sentiments_avg

# Merge daily sentiment with stock data
def merge_sentiment_with_stock(stock_data, sentiment_data):
    sentiment_df = pd.DataFrame(list(sentiment_data.items()), columns=['Date', 'Sentiment'])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    merged_data = pd.merge(stock_data, sentiment_df, on='Date', how='left').fillna(0)
    return merged_data

# Load stock data
stock_data = pd.read_csv('stock_data.csv')  # Replace with your stock data file path
api = initialize_twitter_api()
sentiment_data = fetch_and_analyze_tweets(api, "AAPL", days=30)

# Merge sentiment data with stock data
merged_data = merge_sentiment_with_stock(stock_data, sentiment_data)
print("Merged Data with Sentiment:\n", merged_data.head())
