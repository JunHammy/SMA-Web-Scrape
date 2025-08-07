"""
YouTube Comment Scraper for iPhone Videos
--------------------------------------------

Description:
This script extracts top-level comments from a list of YouTube videos (specified by video IDs).
It uses the YouTube Data API v3 to retrieve comments and saves the results into a CSV file.
This allows for downstream sentiment analysis, topic modeling, or user behavior studies 
related to the "iPhone 17" discussion on YouTube.

Purpose:
- To collect a fixed number of top-level comments per video.
- To enable qualitative or quantitative analysis on YouTube user opinions.

Key Features:
- Reads video IDs from an input text file (`video_ids.txt`).
- Retrieves up to 50 top-level comments per video (customizable).
- Saves all comment data (author, timestamp, likes, and text) into a structured CSV file.

Dependencies:
- googleapiclient: For accessing YouTube Data API.
- csv: For writing output data to CSV.
- dotenv: To securely load the YouTube API key from a `.env` file.
- time: To manage delay between API calls and avoid quota limits.

Environment Variables (required in a .env file):
- YT_API_KEY

Input:
- `video_ids.txt`: A plain text file containing one video ID per line.

Output:
- `youtube_comments.csv`: A CSV file containing extracted comment metadata including:
    - video_id
    - author
    - published_at
    - like_count
    - text

Notes:
- Handles API pagination using `nextPageToken` to retrieve multiple pages of comments.
- Includes error handling to skip problematic videos without crashing the script.
- Adds a 1-second delay between requests to reduce risk of exceeding quota.
- You may adjust the number of comments per video by changing `COMMENTS_PER_VIDEO`.

Usage:
- Ensure the video ID file and API key are correctly configured.
- Run the script directly: `python script_name.py`
"""

import csv
import time
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
API_KEY = os.getenv("YT_API_KEY")
INPUT_FILE = "video_ids.txt"
OUTPUT_FILE = "youtube_comments.csv"
COMMENTS_PER_VIDEO = 50

# YouTube API Client
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_comments(video_id, max_comments=COMMENTS_PER_VIDEO):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )
        response = request.execute()

        while request and len(comments) < max_comments:
            for item in response.get("items", []):
                comment_snippet = item["snippet"]["topLevelComment"]["snippet"]
                comment_data = {
                    "video_id": video_id,
                    "author": comment_snippet.get("authorDisplayName"),
                    "published_at": comment_snippet.get("publishedAt"),
                    "like_count": comment_snippet.get("likeCount"),
                    "text": comment_snippet.get("textDisplay")
                }
                comments.append(comment_data)
                if len(comments) >= max_comments:
                    break

            if "nextPageToken" in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    textFormat="plainText",
                    pageToken=response["nextPageToken"]
                )
                response = request.execute()
            else:
                break
    except Exception as e:
        print(f"[ERROR] Could not fetch comments for {video_id}: {e}")
    return comments

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    all_comments = []

    for idx, video_id in enumerate(video_ids, 1):
        print(f"Fetching comments from video {idx}/{len(video_ids)} — {video_id}")
        comments = get_comments(video_id)
        all_comments.extend(comments)
        time.sleep(1)  # delay to avoid quota abuse

    print(f"Fetched a total of {len(all_comments)} comments from {len(video_ids)} videos.")

    # Save to CSV
    with open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["video_id", "author", "published_at", "like_count", "text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_comments)

    print(f"Saved comments to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
