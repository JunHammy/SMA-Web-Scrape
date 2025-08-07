"""
YouTube iPhone 17 Video Scraper
--------------------------------

Description:
This script uses the YouTube Data API v3 to search for videos related to the keyword "iPhone 17". 
It retrieves metadata from the top search results across multiple pages and saves the data 
into a CSV file and a plain text file for further analysis or comment scraping.

Purpose:
- To collect titles, video IDs, and publish timestamps for YouTube videos discussing "iPhone 17".
- To enable downstream tasks such as comment scraping, sentiment analysis, or content tracking.

Key Features:
- Sends multiple paginated requests (default: 5 pages) to collect up to 250 videos.
- Extracts each video’s ID, title, and published date.
- Saves the full dataset into a CSV file for structured analysis.
- Saves only the video IDs into a `.txt` file for easy use in subsequent scripts.

Dependencies:
- requests: To interact with the YouTube API via HTTP.
- pandas: For storing and exporting video metadata.
- dotenv: To securely load the YouTube API key from a `.env` file.

Environment Variables (required in a .env file):
- YT_API_KEY

Output:
- `iphone17_youtube_videos.csv`: Contains metadata of all matched videos.
- `video_ids.txt`: A plain text list of video IDs (one per line) for further comment extraction.

Notes:
- The script retrieves up to `MAX_RESULTS` videos per page for `PAGES` number of pages.
- Modify `SEARCH_QUERY`, `MAX_RESULTS`, or `PAGES` to adjust the scope of the search.
- Make sure you have enabled the YouTube Data API v3 and that your API key has sufficient quota.
"""

import requests
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("YT_API_KEY")
SEARCH_QUERY = 'iphone 17'
MAX_RESULTS = 50 
PAGES = 5        

all_videos = []
next_page_token = None

for i in range(PAGES):
    print(f"Fetching page {i+1}...")
    url = 'https://www.googleapis.com/youtube/v3/search'
    params = {
        'part': 'snippet',
        'q': SEARCH_QUERY,
        'type': 'video',
        'maxResults': MAX_RESULTS,
        'key': API_KEY,
        'pageToken': next_page_token
    }

    res = requests.get(url, params=params)
    data = res.json()

    for item in data.get('items', []):
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        publish_time = item['snippet']['publishedAt']
        all_videos.append({'video_id': video_id, 'title': title, 'published': publish_time})

    next_page_token = data.get('nextPageToken')
    if not next_page_token:
        break

# Save to CSV
df = pd.DataFrame(all_videos)
df.to_csv("iphone17_youtube_videos.csv", index=False)
print(f"Saved {len(df)} videos to iphone17_youtube_videos.csv")

# Save video IDs for comment scraping
with open("video_ids.txt", "w") as f:
    for v in all_videos:
        f.write(f"{v['video_id']}\n")
