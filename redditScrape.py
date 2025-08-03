import praw
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="iphone16_analysis_script"
)

# Search in r/iphone and other related subreddits
subreddits = ['iphone', 'apple', 'ios', 'technology']
query = "iPhone 16"
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
df.to_csv("reddit_iphone16.csv", index=False)
print("Saved reddit_iphone16.csv with", len(df), "posts")