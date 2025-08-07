# SMA Assignment 2: Sentiment Analysis & Topic Modelling

This project performs **cross-platform sentiment analysis** and **topic modelling** on user-generated content related to upcoming smartphone releases. The pipeline covers data extraction from Reddit and YouTube, followed by text cleaning, classification, and detailed analytical insights.

---

## 📌 Project Goals

1. Extract and aggregate user opinions from Reddit and YouTube (videos and comments).
2. Clean and classify text data into sentiments (positive, neutral, negative).
3. Perform topic modelling using LDA and TF-IDF clustering to identify prevalent discussion points.
4. Compare sentiment and discussion patterns across platforms.
5. Gain insights into public perception of tech product launches.

---

## 🛠️ Scripts Overview & Usage

Below is a step-by-step explanation of all scripts used in this project:

---

### 1. `redditScrape.py`

**Purpose:**  
Scrapes Reddit posts and comments from specified subreddits using the Pushshift and PRAW APIs.

**Features:**
- Fetches post title, author, upvotes, date, and full comment text.

**Usage:**
- Ensure your Reddit API credentials are stored in `.env`.
- Run the script directly:
  ```bash
  python redditScrape.py
  ```

---

### 2. `ytVidScrape.py`

**Purpose:**
Searches YouTube for videos based on a query string (e.g., “iPhone 16 leaks”) and extracts their video IDs.

**Features:**

* Fetches top 20 video IDs using the YouTube Data API v3.
* Saves IDs to `video_ids.txt`.

**Usage:**

* Requires a valid YouTube Data API key in your `.env` file (`YT_API_KEY`).
* Run the script directly:

  ```bash
  python ytVidScrape.py
  ```

---

### 3. `ytCommentsScrape.py`

**Purpose:**
Fetches top-level comments from YouTube videos listed in `video_ids.txt`.

**Features:**

* Extracts author name, timestamp, likes, and comment text.
* Fetches up to 50 comments per video.

**Usage:**

* Ensure `video_ids.txt` is populated from the previous script.
* Run:

  ```bash
  python ytCommentsScrape.py
  ```

---

### 4. `dataCleaning.py`

**Purpose:**
Performs text preprocessing on Reddit and YouTube datasets.

**Cleaning steps:**

* Language detection and filtering (English-only).
* Lowercasing, stopword removal, tokenization.
* Removes URLs, punctuation, emojis, and non-alphabetic content.

**Usage:**

```bash
python dataCleaning.py
```

---

### 5. `dataClassifcation.py`

**Purpose:**
Classifies cleaned text data into **positive**, **negative**, or **neutral** sentiments.

**Tools Used:**

* TextBlob
* VADER SentimentIntensityAnalyzer

**Features:**

* Calculates polarity and compound scores.
* Adds sentiment label to each record.

**Usage:**

```bash
python dataClassifcation.py
```

---

### 6. `analysis.py`

**Purpose:**
Performs exploratory analysis, sentiment visualization, and topic modelling.

**Features:**

* Generates bar plots for sentiment distribution across platforms.
* Word clouds and frequency charts for top words.
* Topic modelling using:

  * Latent Dirichlet Allocation (LDA)
  * TF-IDF with KMeans clustering
* Cross-platform comparisons and feature-level sentiment insights.

**Output:**

* Visualizations displayed via Matplotlib and Seaborn.
* In-notebook or terminal-based insights for further reporting.

**Usage:**

```bash
python analysis.py
```

---

## 📂 Dependencies

Install required Python packages via pip:

```bash
pip install -r requirements.txt
```

Typical libraries include:

* `pandas`, `numpy`, `nltk`, `matplotlib`, `seaborn`
* `textblob`, `vaderSentiment`, `scikit-learn`
* `google-api-python-client`, `dotenv`, `langdetect`

---

## 🔐 Environment Variables

Ensure you have a `.env` file in your project root with the following:

```env
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_user_agent
YT_API_KEY=your_youtube_api_key
```

---

## 📊 Final Deliverables

* Cleaned, classified datasets for Reddit and YouTube.
* Sentiment distributions visualized per platform.
* Top topics and keywords extracted.
* Comparison of public opinion between platforms.

---

## 🧠 Educational Context

This project was completed as part of **IT335C: Social Media Analytics - Assignment 2: Sentiment Analysis and Topic Modelling**.
The focus is on applying NLP techniques and sentiment analysis on social media data to understand public perception of upcoming tech products, specifically new smartphone releases.

---

## 🔄 Recommended Run Order

```bash
python redditScrape.py
python ytVidScrape.py
python ytCommentsScrape.py
python dataCleaning.py
python dataClassifcation.py
python analysis.py
```

---