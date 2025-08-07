"""
Reddit iPhone Posts Scraper
--------------------------------

Description:
This script uses the PRAW (Python Reddit API Wrapper) library to search and scrape Reddit posts 
related to the keyword "iPhone 17" from a list of relevant subreddits. It collects metadata 
from each post and saves the results into a CSV file for later analysis.

Purpose:
- To gather user-generated content (titles, post texts, scores, comment counts, URLs) 
  about the upcoming iPhone 17.
- To perform comparative or sentiment analysis based on Reddit discussion trends.

Key Features:
- Searches in multiple relevant subreddits: 'iphone', 'apple', 'ios', and 'technology'.
- Retrieves up to 100 of the most recent posts per subreddit containing the keyword "iPhone 17".
- Extracts relevant data fields (title, text, score, comment count, URL).
- Outputs a structured CSV file: `reddit_iphone17.csv`.

Dependencies:
- praw: For Reddit API interaction.
- pandas: For data manipulation and saving to CSV.
- dotenv: To securely load Reddit API credentials from a `.env` file.

Environment Variables (required in a .env file):
- REDDIT_CLIENT_ID
- REDDIT_CLIENT_SECRET

Output:
- A CSV file named `reddit_iphone17.csv` saved in the working directory, 
  containing one row per matched Reddit post.

Note:
- Ensure you have a valid Reddit app with proper credentials.
- This script uses a custom user agent: "iphone17_analysis_script".
"""

import praw
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="iphone17_analysis_script"
)

# Search in r/iphone and other related subreddits
subreddits = ['iphone', 'apple', 'ios', 'technology']
query = "iPhone 17"
limit = 100

posts = []

for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    for submission in subreddit.search(query, sort="new", limit=limit):
        posts.append({
            "subreddit": sub,
            "title": submission.title,
            "text": submission.selftext,
            "score": submission.score,
            "num_comments": submission.num_comments,
            "url": submission.url
        })

# Save to CSV
df = pd.DataFrame(posts)
df.to_csv("reddit_iphone17.csv", index=False)
print("Saved reddit_iphone17.csv with", len(df), "posts")