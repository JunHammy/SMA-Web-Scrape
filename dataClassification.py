"""
Comprehensive Social Media Text Preprocessing, Sentiment Analysis & Topic Modeling Pipeline
--------------------------------------------------------------------------------------------

Description:
This script performs a complete pipeline for cleaning, processing, analyzing, and visualizing
social media text data (from Reddit, YouTube, etc.) for sentiment and topic modeling purposes.

It is designed to handle raw CSV datasets, clean them while preserving sentiment-rich information,
analyze public sentiment using both TextBlob and VADER, and extract key discussion topics using
TF-IDF and Latent Dirichlet Allocation (LDA). The script produces both cleaned CSV outputs and
data visualizations to support exploratory data analysis and reporting.

Pipeline Stages:
----------------
1. **Initial Setup**:
   - Imports required libraries (NLTK, VADER, TextBlob, pandas, scikit-learn, etc.)
   - Sets up plotting styles (Seaborn + Matplotlib)

2. **Data Preprocessing & Cleaning**:
   - Auto-detects source format: Reddit posts, YouTube comments, or YouTube metadata
   - Detects and retains only English text (using langdetect and heuristics)
   - Cleans text by:
     - Lowercasing
     - Expanding contractions
     - Handling punctuation (especially sentiment-bearing ones like '!', '?')
     - Removing non-alphanumeric noise while preserving emotive expressions
   - Applies format-specific cleaning strategies based on text length and structure
   - Removes short or irrelevant entries based on customizable thresholds
   - Preserves negation words (e.g., "not", "never") for sentiment clarity

3. **Sentiment Analysis**:
   - Analyzes sentiment using:
     - **TextBlob**: Polarity and subjectivity
     - **VADER**: Compound, positive, negative, and neutral scores
   - Adds sentiment columns to the DataFrame
   - Classifies sentiment into categories: "Positive", "Neutral", "Negative"

4. **Topic Modeling**:
   - Uses both:
     - **TF-IDF Vectorization + KMeans Clustering** for topic grouping
     - **LDA (Latent Dirichlet Allocation)** for probabilistic topic discovery
   - Tokenizes and vectorizes the cleaned text
   - Displays top keywords for each topic (customizable number of topics)

5. **Visualization**:
   - Sentiment Distribution Plot
   - Word Count Distribution
   - Polarity vs Subjectivity Scatterplot
   - Topic Keywords Bar Plots (TF-IDF, LDA)
   - KMeans Topic Cluster Plot (if enabled)

6. **Output & Reporting**:
   - Saves cleaned dataset to `./clean_data/sentiment_ready_<filename>.csv`
   - Generates a summary report CSV with:
     - File name
     - Number of rows before/after cleaning
     - Average text length
   - Optional: Exports topic models and visual plots (customizable)

Input:
- CSV files located in `./data`
- Must contain at least one text field: `text`, `title`, `comment`, `description`, etc.

Output:
- Cleaned and enriched CSV files in `./clean_data`
- Summary report CSV: `sentiment_cleaning_report.csv`
- Visual plots (shown inline or optionally saved)

Key Libraries Used:
- `pandas`, `numpy`, `nltk`, `re`, `matplotlib`, `seaborn`
- `TextBlob`, `vaderSentiment` (for sentiment analysis)
- `sklearn` (for vectorization and KMeans)
- `LatentDirichletAllocation` (from `sklearn.decomposition`)

Assumptions:
- English-based sentiment and topic analysis
- LangDetect may fail on short or malformed text — fallback logic included
- Data is informal (social media-like) with mixed casing, emojis, and punctuation

Usage:
1. Place raw datasets in `./data`
2. Run this script: `python script_name.py`
3. Access cleaned data and reports in `./clean_data`

Example Output Columns:
- `cleaned_text`, `blob_polarity`, `blob_subjectivity`, `vader_compound`, `sentiment_label`

Notes:
- Supports batch processing of multiple CSVs
- Designed for exploratory NLP and sentiment research
- Modular sections can be reused for other preprocessing pipelines

"""

import pandas as pd
import os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Configuration
INPUT_FOLDER = "./clean_data"
OUTPUT_FOLDER = "./classified_data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def classify_sentiment_textblob(text):
    """Classify sentiment using TextBlob (Polarity-based)"""
    if pd.isnull(text) or text == "":
        return "neutral", 0.0
    
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return "positive", polarity
        elif polarity < -0.1:
            return "negative", polarity
        else:
            return "neutral", polarity
    except:
        return "neutral", 0.0

def classify_sentiment_vader(text):
    """Classify sentiment using VADER (Social Media optimized)"""
    if pd.isnull(text) or text == "":
        return "neutral", 0.0
    
    try:
        scores = analyzer.polarity_scores(str(text))
        compound = scores['compound']
        
        if compound >= 0.05:
            return "positive", compound
        elif compound <= -0.05:
            return "negative", compound
        else:
            return "neutral", compound
    except:
        return "neutral", 0.0

class IntegratedTopicClassifier:
    """Integrated sentiment and topic classification system"""
    
    def __init__(self, n_topics=8, n_clusters=6):
        self.n_topics = n_topics
        self.n_clusters = n_clusters
        self.lda_model = None
        self.kmeans_model = None
        self.count_vectorizer = None
        self.tfidf_vectorizer = None
        self.topic_labels = {}
        self.cluster_labels = {}
        
    def extract_advanced_topics(self, texts, product_name):
        """Extract topics using both LDA and K-means clustering"""
        print(f"    Running advanced topic modeling...")
        
        # Prepare texts
        valid_texts = [str(text) for text in texts if pd.notna(text) and str(text).strip() != "" and len(str(text)) > 20]
        
        if len(valid_texts) < max(self.n_topics, self.n_clusters):
            self.n_topics = max(3, len(valid_texts) // 3)
            self.n_clusters = max(3, len(valid_texts) // 3)
            print(f"    Adjusted to {self.n_topics} topics and {self.n_clusters} clusters")
        
        # Method 1: LDA Topic Modeling
        print(f"    LDA topic modeling...")
        self.count_vectorizer = CountVectorizer(
            max_df=0.8,
            min_df=3,
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            doc_term_matrix = self.count_vectorizer.fit_transform(valid_texts)
            
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=50,
                learning_method='online'
            )
            
            lda_topics = self.lda_model.fit_transform(doc_term_matrix)
            
            # Generate LDA topic labels
            self._generate_lda_topic_labels(product_name)
            
        except Exception as e:
            print(f"    LDA error: {e}")
            lda_topics = np.zeros((len(valid_texts), self.n_topics))
        
        # Method 2: TF-IDF + K-means Clustering  
        print(f"    K-means clustering...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=300,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_texts)
            
            self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_assignments = self.kmeans_model.fit_predict(tfidf_matrix)
            
            # Generate cluster labels
            self._generate_cluster_labels(product_name)
            
        except Exception as e:
            print(f"    K-means error: {e}")
            cluster_assignments = np.zeros(len(valid_texts))
        
        return {
            'lda_topics': lda_topics,
            'cluster_assignments': cluster_assignments,
            'valid_texts': valid_texts,
            'topic_labels': self.topic_labels,
            'cluster_labels': self.cluster_labels
        }
    
    def _generate_lda_topic_labels(self, product_name):
        """Generate meaningful labels for LDA topics"""
        if not self.lda_model or not self.count_vectorizer:
            return
            
        feature_names = self.count_vectorizer.get_feature_names_out()
        
        # iPhone-specific topic patterns
        iphone_patterns = {
            'camera_photo': ['camera', 'photo', 'picture', 'video', 'lens', 'portrait', 'night', 'zoom', 'selfie'],
            'performance_speed': ['fast', 'speed', 'performance', 'processor', 'chip', 'lag', 'smooth', 'slow', 'quick'],
            'battery_charging': ['battery', 'charge', 'charging', 'power', 'life', 'wireless', 'dead', 'drain'],
            'design_build': ['design', 'color', 'size', 'weight', 'build', 'material', 'look', 'beautiful', 'ugly'],
            'display_screen': ['screen', 'display', 'oled', 'bright', 'resolution', 'refresh', 'touch', 'crack'],
            'price_value': ['price', 'cost', 'expensive', 'cheap', 'worth', 'value', 'money', 'buy', 'afford'],
            'comparison_vs': ['better', 'worse', 'compare', 'vs', 'android', 'samsung', 'google', 'competition'],
            'software_ios': ['ios', 'software', 'update', 'feature', 'app', 'siri', 'bug', 'glitch'],
        }
        
        for topic_idx in range(self.n_topics):
            top_words_idx = self.lda_model.components_[topic_idx].argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            
            # Match with iPhone patterns
            best_match = "General Discussion"
            max_score = 0
            
            for pattern_name, keywords in iphone_patterns.items():
                score = sum(1 for word in top_words if any(kw in word.lower() for kw in keywords))
                if score > max_score:
                    max_score = score
                    best_match = pattern_name.replace('_', ' & ').title()
            
            # If no strong match, use top words
            if max_score < 2:
                best_match = f"{top_words[0].title()} Discussion"
            
            self.topic_labels[topic_idx] = best_match
    
    def _generate_cluster_labels(self, product_name):
        """Generate meaningful labels for K-means clusters"""
        if not self.kmeans_model or not self.tfidf_vectorizer:
            return
            
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        for cluster_idx in range(self.n_clusters):
            # Get top terms for this cluster
            center = self.kmeans_model.cluster_centers_[cluster_idx]
            top_indices = center.argsort()[-8:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            
            # Create descriptive label
            cluster_label = f"Cluster: {', '.join(top_terms[:3])}"
            self.cluster_labels[cluster_idx] = cluster_label
    
    def assign_topics_to_texts(self, texts):
        """Assign topics to new texts"""
        valid_texts = [str(text) for text in texts if pd.notna(text) and str(text).strip() != ""]
        
        topic_assignments = []
        cluster_assignments = []
        
        if len(valid_texts) == 0:
            return [], []
        
        try:
            # LDA topic assignment
            if self.lda_model and self.count_vectorizer:
                doc_term_matrix = self.count_vectorizer.transform(valid_texts)
                lda_probs = self.lda_model.transform(doc_term_matrix)
                lda_assignments = [np.argmax(probs) for probs in lda_probs]
                topic_assignments = [self.topic_labels.get(idx, f"Topic {idx}") for idx in lda_assignments]
            
            # K-means cluster assignment
            if self.kmeans_model and self.tfidf_vectorizer:
                tfidf_matrix = self.tfidf_vectorizer.transform(valid_texts)
                cluster_ids = self.kmeans_model.predict(tfidf_matrix)
                cluster_assignments = [self.cluster_labels.get(idx, f"Cluster {idx}") for idx in cluster_ids]
                
        except Exception as e:
            print(f"    Topic assignment error: {e}")
            topic_assignments = ["General"] * len(valid_texts)
            cluster_assignments = ["Cluster 0"] * len(valid_texts)
        
        return topic_assignments, cluster_assignments

def categorize_by_themes(text, company_name, product_name):
    """Enhanced theme categorization with predefined categories"""
    if pd.isnull(text) or text == "":
        return ["general"]
    
    text_lower = str(text).lower()
    categories = []
    
    # Enhanced theme keywords for iPhone analysis
    theme_keywords = {
        "product_features": ["feature", "function", "design", "interface", "usability", "performance", "speed", "quality", "camera", "battery"],
        "pricing": ["price", "cost", "expensive", "cheap", "value", "money", "afford", "budget", "free", "subscription", "worth"],
        "customer_service": ["service", "support", "help", "staff", "representative", "response", "assistance", "agent", "customer"],
        "user_experience": ["experience", "easy", "difficult", "intuitive", "confusing", "smooth", "frustrating", "user friendly", "interface"],
        "comparison": ["better", "worse", "compare", "vs", "versus", "competitor", "alternative", "similar", "android", "samsung"],
        "technical_issues": ["bug", "error", "crash", "problem", "issue", "broken", "fix", "glitch", "update", "software"],
        "delivery_shipping": ["delivery", "shipping", "fast", "slow", "arrived", "package", "order", "logistics", "store"],
        "brand_perception": ["brand", "company", "trust", "reputation", "reliable", "professional", "image", "apple"]
    }
    
    # Check for theme keywords
    for theme, keywords in theme_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            categories.append(theme)
    
    # Check for company/product specific mentions
    if company_name.lower() in text_lower or product_name.lower() in text_lower:
        categories.append("brand_mention")
    
    # If no categories found, assign "general"
    if not categories:
        categories = ["general"]
    
    return categories

def classify_dataset_integrated(df, company_name, product_name):
    """Integrated classification with advanced topic modeling"""
    print(f"  Running integrated sentiment & topic classification...")
    
    # Find text columns to analyze
    text_columns = [col for col in df.columns if col.startswith('clean_')]
    
    if not text_columns:
        print("  No cleaned text columns found!")
        return df
    
    # Initialize topic classifier
    topic_classifier = IntegratedTopicClassifier(n_topics=6, n_clusters=5)
    
    # For each text column, perform comprehensive classification
    for text_col in text_columns:
        base_name = text_col.replace('clean_', '')
        
        print(f"    Processing column: {text_col}")
        
        # 1. SENTIMENT ANALYSIS
        # TextBlob sentiment
        sentiment_data = df[text_col].apply(classify_sentiment_textblob)
        df[f'{base_name}_sentiment_textblob'] = [s[0] for s in sentiment_data]
        df[f'{base_name}_polarity_textblob'] = [s[1] for s in sentiment_data]
        
        # VADER sentiment
        sentiment_data_vader = df[text_col].apply(classify_sentiment_vader)
        df[f'{base_name}_sentiment_vader'] = [s[0] for s in sentiment_data_vader]
        df[f'{base_name}_compound_vader'] = [s[1] for s in sentiment_data_vader]
        
        # Combined sentiment
        def combine_sentiments(tb_sent, vader_sent):
            if tb_sent == vader_sent:
                return tb_sent
            else:
                return "neutral"
        
        df[f'{base_name}_sentiment_combined'] = df.apply(
            lambda row: combine_sentiments(
                row[f'{base_name}_sentiment_textblob'], 
                row[f'{base_name}_sentiment_vader']
            ), axis=1
        )
        
        # 2. PREDEFINED THEME CATEGORIZATION
        df[f'{base_name}_themes'] = df[text_col].apply(
            lambda x: categorize_by_themes(x, company_name, product_name)
        )
        
        # 3. ADVANCED TOPIC MODELING
        print(f"    Advanced topic modeling for {text_col}...")
        topic_results = topic_classifier.extract_advanced_topics(df[text_col].values, product_name)
        
        # Assign LDA topics and clusters to all texts
        lda_topics, cluster_assignments = topic_classifier.assign_topics_to_texts(df[text_col].values)
        
        # Add topic assignments to dataframe
        if len(lda_topics) == len(df):
            df[f'{base_name}_lda_topic'] = lda_topics
            df[f'{base_name}_kmeans_cluster'] = cluster_assignments
        else:
            # Handle size mismatch (due to text filtering)
            df[f'{base_name}_lda_topic'] = ["General"] * len(df)
            df[f'{base_name}_kmeans_cluster'] = ["Cluster 0"] * len(df)
        
        # Store topic information
        df.attrs[f'{base_name}_topic_info'] = {
            'lda_labels': topic_classifier.topic_labels,
            'cluster_labels': topic_classifier.cluster_labels,
            'n_topics': topic_classifier.n_topics,
            'n_clusters': topic_classifier.n_clusters
        }
        
        # Print discovered topics
        print(f"    Discovered LDA Topics:")
        for idx, label in topic_classifier.topic_labels.items():
            print(f"      Topic {idx}: {label}")
    
    print(f"  Classification complete with integrated topic modeling!")
    return df

def generate_enhanced_classification_summary(df, filename):
    """Generate enhanced summary with topic information"""
    summary = {
        'filename': filename,
        'total_records': len(df),
    }
    
    # Sentiment distribution
    sentiment_columns = [col for col in df.columns if 'sentiment_combined' in col]
    if sentiment_columns:
        for col in sentiment_columns:
            sentiment_dist = df[col].value_counts().to_dict()
            summary[f'{col}_distribution'] = sentiment_dist
    
    # Traditional theme distribution
    theme_columns = [col for col in df.columns if col.endswith('_themes')]
    if theme_columns:
        all_themes = []
        for col in theme_columns:
            for theme_list in df[col].dropna():
                if isinstance(theme_list, list):
                    all_themes.extend(theme_list)
                else:
                    all_themes.append(str(theme_list))
        
        theme_counts = Counter(all_themes)
        summary['theme_distribution'] = dict(theme_counts.most_common(10))
    
    # LDA topic distribution
    lda_columns = [col for col in df.columns if col.endswith('_lda_topic')]
    if lda_columns:
        for col in lda_columns:
            lda_dist = df[col].value_counts().to_dict()
            summary[f'{col}_distribution'] = lda_dist
    
    # K-means cluster distribution
    cluster_columns = [col for col in df.columns if col.endswith('_kmeans_cluster')]
    if cluster_columns:
        for col in cluster_columns:
            cluster_dist = df[col].value_counts().to_dict()
            summary[f'{col}_distribution'] = cluster_dist
    
    return summary

def main():
    print("INTEGRATED SENTIMENT & TOPIC CLASSIFICATION")
    print("=" * 70)
    print("Features: Sentiment Analysis + Manual Themes + LDA Topics + K-means Clusters")
    print("=" * 70)
    
    # Get company and product information
    company_name = input("Enter the company name: ").strip()
    product_name = input("Enter the product/service name: ").strip()
    
    print(f"\nAnalyzing sentiment and topics for: {company_name} - {product_name}")
    print("=" * 70)
    
    all_summaries = []
    
    # Process each cleaned file
    for filename in os.listdir(INPUT_FOLDER):
        if filename.startswith("sentiment_ready_") and filename.endswith(".csv"):
            print(f"\nProcessing: {filename}")
            filepath = os.path.join(INPUT_FOLDER, filename)
            
            try:
                # Load cleaned data
                df = pd.read_csv(filepath)
                print(f"  Loaded {len(df)} records")
                
                # Perform integrated classification
                df_classified = classify_dataset_integrated(df, company_name, product_name)
                
                # Generate enhanced summary
                summary = generate_enhanced_classification_summary(df_classified, filename)
                all_summaries.append(summary)
                
                # Save classified data
                output_filename = filename.replace("sentiment_ready_", "integrated_classified_")
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                df_classified.to_csv(output_path, index=False)
                print(f"  Saved classified data to: {output_path}")
                
                # Display sample results
                print(f"  Sample Results:")
                sentiment_cols = [col for col in df_classified.columns if 'sentiment_combined' in col]
                if sentiment_cols:
                    sample_sentiments = df_classified[sentiment_cols[0]].value_counts()
                    for sent, count in sample_sentiments.head().items():
                        print(f"    Sentiment - {sent}: {count} ({count/len(df_classified)*100:.1f}%)")
                
                # Show topic distribution
                lda_cols = [col for col in df_classified.columns if 'lda_topic' in col]
                if lda_cols:
                    print(f"    Top LDA Topics:")
                    top_topics = df_classified[lda_cols[0]].value_counts().head(3)
                    for topic, count in top_topics.items():
                        print(f"      {topic}: {count}")
                
            except Exception as e:
                print(f"  ERROR processing {filename}: {str(e)}")
                continue
    
    # Generate overall summary report
    print("\n" + "=" * 70)
    print("INTEGRATED CLASSIFICATION SUMMARY")
    print("=" * 70)
    
    if all_summaries:
        total_records = sum(s['total_records'] for s in all_summaries)
        print(f"Total records classified: {total_records:,}")
        print(f"Methods used: Sentiment Analysis + Manual Themes + LDA + K-means")
        
        # Save summary report
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(OUTPUT_FOLDER, "integrated_classification_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary report saved to: {summary_path}")
    
    print(f"\nIntegrated classified files saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()