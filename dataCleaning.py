"""
Sentiment Analysis Preprocessing Pipeline
-----------------------------------------

Description:
This script performs advanced data cleaning and preparation for sentiment analysis on datasets
containing text from various sources (Reddit posts, YouTube comments, YouTube video titles/descriptions).
It focuses on:
- Language detection (to keep only English content)
- Preservation of sentiment-bearing text (e.g., exclamations, negations)
- Text normalization and filtering
- Format detection (Reddit, YouTube, etc.) for tailored cleaning logic
- Generating cleaned output and a summary report

Purpose:
- To transform raw social media and video platform data into sentiment-analysis-ready format.
- To improve the quality of sentiment insights by filtering out irrelevant, non-English, or meaningless content.

Key Features:
- **Language Detection**: Uses `langdetect` and heuristics to retain only English content.
- **Sentiment-Preserving Cleaning**: Carefully handles contractions, punctuation, and negations.
- **Stopword Filtering**: Removes general stopwords while retaining sentiment-bearing ones like "not", "never", "so", etc.
- **Format-Aware Cleaning**: Automatically detects the type of dataset (Reddit, YouTube comments/videos) and applies relevant cleaning rules.
- **Content Length Filtering**: Removes short texts that are insufficient for sentiment analysis.
- **Reporting**: Generates both console output and a detailed CSV summary report with statistics on text retention and average lengths.

Input:
- CSV files located in the `./data` directory.
- Each file must have at least one text-based column such as `title`, `text`, `comment`, or `description`.

Output:
- Cleaned CSV files saved in the `./clean_data` directory, prefixed with `sentiment_ready_`.
- A report file `sentiment_cleaning_report.csv` summarizing the number of rows filtered and average text length per file.

Dependencies:
- `pandas`, `nltk`, `re`, `langdetect`, `os`, `dotenv`
- Required NLTK downloads: "punkt", "stopwords"

Environment Variables (expected in a `.env` file):
- None specifically required for this script, but it uses `.env` structure in case expansion is needed.

Assumptions:
- Input data is in CSV format.
- LangDetect can fail on short or malformed text, in which case fallback heuristics are applied.
- Cleaning is optimized for English sentiment tasks (e.g., social media sentiment mining, topic detection).

Usage:
1. Place input `.csv` files in the `./data` folder.
2. Run this script: `python script_name.py`
3. Find cleaned data and summary report in the `./clean_data` folder.

This pipeline helps ensure that the final dataset is both semantically rich and suitable for downstream
NLP tasks like sentiment classification or topic modeling.
"""

import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect

# Download NLTK resources (only runs once)
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# File paths
INPUT_FOLDER = "./data"
OUTPUT_FOLDER = "./clean_data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# English stopwords (but we'll be more selective for sentiment analysis)
stop_words = set(stopwords.words("english"))

# Remove sentiment-bearing words from stopwords for sentiment analysis
sentiment_bearing_stopwords = {
    'not', 'no', 'nor', 'but', 'however', 'although', 'though', 'yet', 'except',
    'nevertheless', 'nonetheless', 'despite', 'in spite of', 'rather', 'instead',
    'otherwise', 'else', 'only', 'just', 'even', 'still', 'already', 'always',
    'never', 'ever', 'very', 'really', 'quite', 'too', 'so', 'such', 'much',
    'many', 'more', 'most', 'less', 'least', 'few', 'little', 'enough'
}

# Adjust stopwords for sentiment analysis
filtered_stop_words = stop_words - sentiment_bearing_stopwords

def detect_language(text):
    """Detect if text is in English"""
    if pd.isnull(text) or len(str(text).strip()) < 10:
        return False
    
    try:
        detected_lang = detect(str(text))
        return detected_lang == 'en'
    except Exception:
        # Fallback: check for common English words
        text_lower = str(text).lower()
        english_indicators = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use']
        english_word_count = sum(1 for word in english_indicators if word in text_lower)
        return english_word_count >= 2

def clean_text_for_sentiment(text):
    """Clean text specifically for sentiment analysis - preserve sentiment-bearing elements"""
    if pd.isnull(text) or text == "" or text.lower() in ['nan', 'none', 'null']:
        return ""
    
    text = str(text)
    
    # Store original for language detection
    original_text = text
    
    # Remove URLs but preserve the context
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' [URL] ', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' [URL] ', text)
    
    # Remove Reddit-specific patterns but keep context
    text = re.sub(r'/r/\w+', ' [SUBREDDIT] ', text)
    text = re.sub(r'/u/\w+', ' [USER] ', text)
    text = re.sub(r'u/\w+', ' [USER] ', text)
    text = re.sub(r'r/\w+', ' [SUBREDDIT] ', text)
    
    # Remove social media handles but preserve mentions context
    text = re.sub(r'@\w+', ' [MENTION] ', text)
    text = re.sub(r'#\w+', ' [HASHTAG] ', text)
    
    # Preserve important punctuation for sentiment (! ? ...)
    # Convert multiple exclamation/question marks to single ones
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\.{3,}', '...', text)
    
    # Handle contractions (important for sentiment)
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
        "'m": " am", "it's": "it is", "that's": "that is",
        "what's": "what is", "where's": "where is", "how's": "how is",
        "here's": "here is", "there's": "there is", "who's": "who is"
    }
    
    for contraction, expansion in contractions.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    
    # Remove excessive special characters but preserve sentiment indicators
    text = re.sub(r'[^\w\s!?.,;:\-\'"()]', ' ', text)
    
    # Handle numbers - remove standalone numbers but keep dates/versions (iPhone 17, iOS 18, etc.)
    text = re.sub(r'\b\d+\b(?!\s*(pro|max|plus|mini|iphone|ios|version))', '', text, flags=re.IGNORECASE)
    
    # Convert to lowercase but preserve sentence structure
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Tokenize and filter stopwords (but keep sentiment-bearing ones)
    try:
        tokens = word_tokenize(text)
        # Keep words that are: not in filtered stopwords, longer than 1 char, or important punctuation
        filtered_tokens = []
        for token in tokens:
            if (token not in filtered_stop_words and len(token) > 1) or token in ['!', '?', '...']:
                filtered_tokens.append(token)
        
        return " ".join(filtered_tokens)
    except:
        # Fallback if tokenization fails
        words = text.split()
        words = [word for word in words if word not in filtered_stop_words and len(word) > 1]
        return " ".join(words)

def detect_data_format(df):
    """Detect the format of the data based on column names"""
    columns = df.columns.tolist()
    
    if 'subreddit' in columns and 'title' in columns:
        return 'reddit_posts'
    elif 'video_id' in columns and 'text' in columns:
        return 'youtube_comments'
    elif 'title' in columns and 'description' in columns:
        return 'youtube_videos'
    else:
        return 'unknown'

def clean_dataset_for_sentiment(filename, df):
    """Clean dataset specifically for sentiment analysis"""
    data_format = detect_data_format(df)
    print(f"  Detected format: {data_format}")
    
    original_count = len(df)
    
    if data_format == 'reddit_posts':
        # Process title and text columns
        if 'title' in df.columns:
            # Check language for titles
            print("  Filtering English titles...")
            df['is_english_title'] = df['title'].apply(detect_language)
            
            df['clean_title'] = df['title'].apply(clean_text_for_sentiment)
            df['title_length'] = df['clean_title'].str.len()
        
        if 'text' in df.columns:
            # Check language for text content
            print("  Filtering English text content...")
            df['is_english_text'] = df['text'].apply(detect_language)
            
            df['clean_text'] = df['text'].apply(clean_text_for_sentiment)
            df['text_length'] = df['clean_text'].str.len()
        
        # Filter for English content only
        if 'is_english_title' in df.columns and 'is_english_text' in df.columns:
            df = df[(df['is_english_title'] == True) | (df['is_english_text'] == True)]
        elif 'is_english_title' in df.columns:
            df = df[df['is_english_title'] == True]
        elif 'is_english_text' in df.columns:
            df = df[df['is_english_text'] == True]
        
        # Filter out very short content (insufficient for sentiment analysis)
        if 'clean_title' in df.columns:
            df = df[df['title_length'] > 15]  # Minimum for meaningful sentiment
        
    elif data_format == 'youtube_comments':
        if 'text' in df.columns:
            print("  Filtering English comments...")
            df['is_english'] = df['text'].apply(detect_language)
            df = df[df['is_english'] == True]
            
            df['clean_text'] = df['text'].apply(clean_text_for_sentiment)
            df['text_length'] = df['clean_text'].str.len()
            
            # Remove very short comments (insufficient for sentiment analysis)
            df = df[df['text_length'] > 10]
            
            # Remove comments that are just URLs or mentions
            df = df[~df['clean_text'].str.contains(r'^\s*\[URL\]\s*$|^\s*\[MENTION\]\s*$|^\s*\[HASHTAG\]\s*$')]
    
    elif data_format == 'youtube_videos':
        if 'title' in df.columns:
            print("  Filtering English video titles...")
            df['is_english_title'] = df['title'].apply(detect_language)
            df = df[df['is_english_title'] == True]
            df['clean_title'] = df['title'].apply(clean_text_for_sentiment)
            
        if 'description' in df.columns:
            print("  Filtering English video descriptions...")
            df['is_english_desc'] = df['description'].apply(detect_language)
            df['clean_description'] = df['description'].apply(clean_text_for_sentiment)
    
    # Generic cleaning for unknown formats
    else:
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if any(keyword in col.lower() for keyword in ['text', 'title', 'body', 'content', 'comment']):
                print(f"  Filtering English content in column: {col}")
                df[f'is_english_{col}'] = df[col].apply(detect_language)
                df[f'clean_{col}'] = df[col].apply(clean_text_for_sentiment)
    
    # Remove rows where all text became empty after cleaning
    text_cols = [col for col in df.columns if col.startswith('clean_')]
    if text_cols:
        # Keep rows where at least one clean text column has content
        df['has_content'] = df[text_cols].apply(lambda row: any(str(val).strip() != '' for val in row), axis=1)
        df = df[df['has_content'] == True]
        df = df.drop('has_content', axis=1)
    
    filtered_count = len(df)
    print(f"  Language filtering: {original_count} -> {filtered_count} rows ({((original_count - filtered_count) / original_count * 100):.1f}% removed)")
    
    return df

def generate_sentiment_report(original_df, cleaned_df, filename):
    """Generate a report specifically for sentiment analysis preparation"""
    # Count English vs non-English content
    english_cols = [col for col in cleaned_df.columns if col.startswith('is_english')]
    
    report = {
        'filename': filename,
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100 if len(original_df) > 0 else 0,
        'english_detection_columns': len(english_cols)
    }
    
    # Add average text length for sentiment analysis quality
    clean_text_cols = [col for col in cleaned_df.columns if col.startswith('clean_') and 'length' not in col]
    if clean_text_cols:
        avg_lengths = {}
        for col in clean_text_cols:
            lengths = cleaned_df[col].str.len().dropna()
            if len(lengths) > 0:
                avg_lengths[col] = lengths.mean()
        report['average_text_lengths'] = avg_lengths
    
    return report

def main():
    print("Starting SENTIMENT ANALYSIS data cleaning process...")
    print("Focus: English language detection and sentiment preservation")
    print("=" * 60)
    
    # Check if langdetect is installed
    try:
        from langdetect import detect
        print("Language detection enabled")
    except ImportError:
        print("WARNING: langdetect not installed. Install with: pip install langdetect")
        print("  Falling back to basic English detection")
    
    all_reports = []
    
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".csv"):
            print(f"\nProcessing: {filename}")
            filepath = os.path.join(INPUT_FOLDER, filename)
            
            try:
                # Read the original data
                df_original = pd.read_csv(filepath)
                print(f"  Original shape: {df_original.shape}")
                
                # Clean the dataset for sentiment analysis
                df_cleaned = clean_dataset_for_sentiment(filename, df_original.copy())
                print(f"  Final shape: {df_cleaned.shape}")
                
                # Generate report
                report = generate_sentiment_report(df_original, df_cleaned, filename)
                all_reports.append(report)
                
                # Save cleaned data
                output_path = os.path.join(OUTPUT_FOLDER, f"sentiment_ready_{filename}")
                df_cleaned.to_csv(output_path, index=False)
                print(f"  Saved to: {output_path}")
                
                # Print sample of cleaned data for sentiment analysis
                if len(df_cleaned) > 0:
                    clean_columns = [col for col in df_cleaned.columns if col.startswith('clean_')]
                    if clean_columns:
                        print(f"  Sample cleaned text for sentiment analysis:")
                        sample_texts = df_cleaned[clean_columns[0]].dropna().head(3).values
                        for i, text in enumerate(sample_texts[:2]):
                            if text and len(str(text)) > 10:
                                print(f"    {i+1}. '{str(text)[:80]}...'")
                
            except Exception as e:
                print(f"  ERROR processing {filename}: {str(e)}")
                continue
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS CLEANING SUMMARY")
    print("=" * 60)
    
    if all_reports:
        total_original = sum(r['original_rows'] for r in all_reports)
        total_cleaned = sum(r['cleaned_rows'] for r in all_reports)
        
        for report in all_reports:
            print(f"\n{report['filename']}:")
            print(f"  Original rows: {report['original_rows']:,}")
            print(f"  English rows: {report['cleaned_rows']:,}")
            print(f"  Filtered out: {report['rows_removed']:,} ({report['removal_percentage']:.1f}%)")
            if 'average_text_lengths' in report:
                print(f"  Avg text length: {list(report['average_text_lengths'].values())[0]:.0f} chars")
        
        print(f"\nOVERALL RESULTS:")
        print(f"  Total original: {total_original:,}")
        print(f"  Total English: {total_cleaned:,}")
        print(f"  Retention rate: {(total_cleaned/total_original)*100:.1f}%")
        print(f"  Ready for sentiment analysis: {total_cleaned:,} records")
    
    # Save detailed report
    if all_reports:
        report_df = pd.DataFrame(all_reports)
        report_path = os.path.join(OUTPUT_FOLDER, "sentiment_cleaning_report.csv")
        report_df.to_csv(report_path, index=False)
        print(f"\nDetailed report saved to: {report_path}")
    
    print(f"\nSentiment-ready files saved in: {OUTPUT_FOLDER}")
    
if __name__ == "__main__":
    main()