import pandas as pd
import os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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

def extract_themes_keywords(texts, n_clusters=5, max_features=100):
    """Extract themes/topics from text using TF-IDF and clustering"""
    # Remove empty texts
    valid_texts = [str(text) for text in texts if pd.notna(text) and str(text).strip() != ""]
    
    if len(valid_texts) < n_clusters:
        n_clusters = max(2, len(valid_texts) // 2)
    
    try:
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include both single words and bigrams
            min_df=2,  # Must appear in at least 2 documents
            max_df=0.8  # Must not appear in more than 80% of documents
        )
        
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract top terms for each cluster
        themes = {}
        for i in range(n_clusters):
            # Get top terms for this cluster
            center = kmeans.cluster_centers_[i]
            top_indices = center.argsort()[-10:][::-1]  # Top 10 terms
            top_terms = [feature_names[idx] for idx in top_indices]
            themes[f"theme_{i+1}"] = top_terms
        
        return themes, clusters, vectorizer
    
    except Exception as e:
        print(f"Theme extraction error: {e}")
        return {}, [], None

def categorize_by_themes(text, company_name, product_name):
    """Categorize text based on predefined themes relevant to the company/product"""
    if pd.isnull(text) or text == "":
        return ["general"]
    
    text_lower = str(text).lower()
    categories = []
    
    # Define theme keywords (customize based on your company/product)
    theme_keywords = {
        "product_features": ["feature", "function", "design", "interface", "usability", "performance", "speed", "quality"],
        "pricing": ["price", "cost", "expensive", "cheap", "value", "money", "afford", "budget", "free", "subscription"],
        "customer_service": ["service", "support", "help", "staff", "representative", "response", "assistance", "agent"],
        "user_experience": ["experience", "easy", "difficult", "intuitive", "confusing", "smooth", "frustrating", "user friendly"],
        "comparison": ["better", "worse", "compare", "vs", "versus", "competitor", "alternative", "similar"],
        "technical_issues": ["bug", "error", "crash", "problem", "issue", "broken", "fix", "glitch", "update"],
        "delivery_shipping": ["delivery", "shipping", "fast", "slow", "arrived", "package", "order", "logistics"],
        "brand_perception": ["brand", "company", "trust", "reputation", "reliable", "professional", "image"]
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

def classify_dataset(df, company_name, product_name):
    """Classify entire dataset with sentiment and themes"""
    print(f"  Classifying sentiment and themes...")
    
    # Find text columns to analyze
    text_columns = [col for col in df.columns if col.startswith('clean_')]
    
    if not text_columns:
        print("  No cleaned text columns found!")
        return df
    
    # For each text column, perform classification
    for text_col in text_columns:
        base_name = text_col.replace('clean_', '')
        
        print(f"    Processing column: {text_col}")
        
        # Sentiment Analysis - TextBlob
        sentiment_data = df[text_col].apply(classify_sentiment_textblob)
        df[f'{base_name}_sentiment_textblob'] = [s[0] for s in sentiment_data]
        df[f'{base_name}_polarity_textblob'] = [s[1] for s in sentiment_data]
        
        # Sentiment Analysis - VADER
        sentiment_data_vader = df[text_col].apply(classify_sentiment_vader)
        df[f'{base_name}_sentiment_vader'] = [s[0] for s in sentiment_data_vader]
        df[f'{base_name}_compound_vader'] = [s[1] for s in sentiment_data_vader]
        
        # Combined sentiment (majority vote)
        def combine_sentiments(tb_sent, vader_sent):
            if tb_sent == vader_sent:
                return tb_sent
            else:
                return "neutral"  # Default to neutral if disagree
        
        df[f'{base_name}_sentiment_combined'] = df.apply(
            lambda row: combine_sentiments(
                row[f'{base_name}_sentiment_textblob'], 
                row[f'{base_name}_sentiment_vader']
            ), axis=1
        )
        
        # Theme categorization
        df[f'{base_name}_themes'] = df[text_col].apply(
            lambda x: categorize_by_themes(x, company_name, product_name)
        )
    
    # Extract overall themes using clustering (for the first text column)
    if text_columns:
        main_text_col = text_columns[0]
        print(f"    Extracting themes from {main_text_col}...")
        
        themes, clusters, vectorizer = extract_themes_keywords(df[main_text_col].values)
        
        # Add cluster assignments
        if len(clusters) == len(df):
            df['auto_theme_cluster'] = clusters
        
        # Store themes for reporting
        df.attrs['extracted_themes'] = themes
    
    return df

def generate_classification_summary(df, filename):
    """Generate summary statistics for classification"""
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
    
    # Theme distribution
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
    
    return summary

def main():
    print("Starting DATA CLASSIFICATION for Sentiment Analysis")
    print("=" * 60)
    
    # Get company and product information
    company_name = input("Enter the company name: ").strip()
    product_name = input("Enter the product/service name: ").strip()
    
    print(f"\nAnalyzing sentiment and themes for: {company_name} - {product_name}")
    print("=" * 60)
    
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
                
                # Perform classification
                df_classified = classify_dataset(df, company_name, product_name)
                
                # Generate summary
                summary = generate_classification_summary(df_classified, filename)
                all_summaries.append(summary)
                
                # Save classified data
                output_filename = filename.replace("sentiment_ready_", "classified_")
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                df_classified.to_csv(output_path, index=False)
                print(f"  Saved classified data to: {output_path}")
                
                # Display sample results
                print(f"  Sample sentiment distribution:")
                sentiment_cols = [col for col in df_classified.columns if 'sentiment_combined' in col]
                if sentiment_cols:
                    sample_sentiments = df_classified[sentiment_cols[0]].value_counts()
                    for sent, count in sample_sentiments.head().items():
                        print(f"    {sent}: {count} ({count/len(df_classified)*100:.1f}%)")
                
            except Exception as e:
                print(f"  ERROR processing {filename}: {str(e)}")
                continue
    
    # Generate overall summary report
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    
    if all_summaries:
        total_records = sum(s['total_records'] for s in all_summaries)
        print(f"Total records classified: {total_records:,}")
        
        # Save summary report
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(OUTPUT_FOLDER, "classification_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary report saved to: {summary_path}")
    
    print(f"\nClassified files saved in: {OUTPUT_FOLDER}")
    print("\nNext steps:")
    print("1. Review the classification results in the output files")
    print("2. Proceed with detailed sentiment analysis and visualization")
    print("3. Use the classified data for insights and recommendations")

if __name__ == "__main__":
    main()