import csv
import time
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
load_dotenv()

# === CONFIGURATION ===
API_KEY = os.getenv("YT_API_KEY")
INPUT_FILE = "video_ids.txt"
OUTPUT_FILE = "youtube_comments.csv"
COMMENTS_PER_VIDEO = 50

# === SETUP YOUTUBE API CLIENT ===
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
        time.sleep(1)  # polite delay to avoid quota abuse

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
