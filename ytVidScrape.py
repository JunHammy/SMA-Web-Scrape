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
