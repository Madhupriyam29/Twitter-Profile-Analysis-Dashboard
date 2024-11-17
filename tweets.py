import streamlit as st
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import plotly.express as px
from wordcloud import WordCloud
import plotly.graph_objects as go
from collections import Counter
from nltk.corpus import stopwords

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Setup Chrome WebDriver for Selenium
options = Options()
options.add_argument("--headless")  # Run Chrome in headless mode
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Get stopwords from nltk library (additionally, you can define your own custom stopword list)
stop_words = set(stopwords.words('english'))

def scrape_twitter_account(username, scrolls=5, max_tweets=50):
    """Scrapes tweets from the given Twitter username."""
    url = f'https://twitter.com/{username}'
    driver.get(url)
    time.sleep(3)  # Let the page load initially

    # Wait for tweets to load
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//article[@role='article']")))
    except:
        print("Error: Tweets not loaded in time.")
    
    tweet_data = []
    previous_tweet_count = 0
    
    while len(tweet_data) < max_tweets:
        # Scroll to load more tweets
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(3)  # Wait for new tweets to load
        
        # Get page source and parse it
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        tweets = soup.find_all('article', {'role': 'article'})
        
        # Extract tweets content
        for tweet in tweets:
            content = tweet.find('div', {'lang': True})
            if content:
                tweet_text = content.get_text()
                if tweet_text not in tweet_data:  # Avoid duplicates
                    tweet_data.append(tweet_text)

        if len(tweet_data) == previous_tweet_count:
            break

        previous_tweet_count = len(tweet_data)

    return tweet_data

def clean_tweet(tweet):
    """Cleans the tweet text by removing URLs, mentions, and non-alphanumeric characters."""
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)  # Remove URLs
    tweet = re.sub(r"@\w+", "", tweet)  # Remove mentions
    tweet = re.sub(r"#\w+", "", tweet)  # Remove hashtags
    tweet = re.sub(r"[^a-zA-Z0-9\s]", "", tweet)  # Remove non-alphanumeric characters
    tweet = " ".join(tweet.split())  # Remove extra spaces
    return tweet

def analyze_sentiment(tweet):
    """Analyzes sentiment of the tweet using VADER sentiment analyzer."""
    sentiment_score = analyzer.polarity_scores(tweet)
    compound_score = sentiment_score['compound']
    
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def create_sentiment_pie_chart(sentiments):
    """Creates a pie chart for sentiment distribution."""
    sentiment_counts = pd.Series(sentiments).value_counts()
    fig = px.pie(
        sentiment_counts, 
        values=sentiment_counts, 
        names=sentiment_counts.index,
        title='Tweet Sentiment Distribution'
    )
    return fig

def create_wordcloud(tweets):
    """Generates a wordcloud from tweet content."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tweets))
    fig = go.Figure(go.Image(z=wordcloud.to_array()))
    fig.update_layout(title='Word Cloud of Tweets')
    return fig

def extract_negative_words(tweets):
    """Extracts the most common negative words from negative tweets."""
    negative_words = []
    
    for tweet in tweets:
        sentiment = analyze_sentiment(tweet)
        if sentiment == "Negative":
            # Clean tweet and split into words
            cleaned_tweet = clean_tweet(tweet)
            words = cleaned_tweet.split()
            # Remove stopwords (common words like 'the', 'is', 'in', etc.)
            meaningful_words = [word for word in words if word.lower() not in stop_words]
            negative_words.extend(meaningful_words)
    
    return negative_words

def get_most_common_negative_words(negative_words, num_words=10):
    """Counts and returns the most common negative words."""
    word_counts = Counter(negative_words)
    common_negative_words = word_counts.most_common(num_words)
    return common_negative_words

# Streamlit App layout
st.title("Twitter Profile Analysis Dashboard")

# Input for Twitter username
username = st.text_input("Enter Twitter Username (without '@'): ")

# Button to trigger scraping
if st.button("Scrape Tweets"):
    if username:
        st.write(f"Scraping tweets from {username}...")
        
        # Scrape tweets from the Twitter account
        tweets = scrape_twitter_account(username, scrolls=5, max_tweets=50)
        
        if len(tweets) == 0:
            st.write("No tweets found or unable to fetch data.")
        else:
            st.write(f"Fetched {len(tweets)} tweets from @{username}")

            # Perform sentiment analysis on the tweets
            sentiments = [analyze_sentiment(tweet) for tweet in tweets]

            # Display sentiment pie chart
            sentiment_fig = create_sentiment_pie_chart(sentiments)
            st.plotly_chart(sentiment_fig)

            # Generate word cloud for all tweets
            cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]
            wordcloud_fig = create_wordcloud(cleaned_tweets)
            st.plotly_chart(wordcloud_fig)

            # Extract and display most common negative words
            negative_words = extract_negative_words(tweets)
            common_negative_words = get_most_common_negative_words(negative_words)

            if common_negative_words:
                # Display the most common negative words
                common_negative_words_df = pd.DataFrame(common_negative_words, columns=["Word", "Frequency"])
                st.subheader("Most Common Negative Words in Tweets:")
                st.dataframe(common_negative_words_df)
            else:
                st.write("No negative words found in the tweets.")

            # Show the latest tweets
            st.subheader("Latest Tweets:")
            for idx, tweet in enumerate(tweets, 1):
                st.write(f"{idx}. {tweet}")
    else:
        st.write("Please enter a valid username.")

# Close the WebDriver after usage
driver.quit()