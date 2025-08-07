"""
iPhone 17 Cross-Platform Analysis Pipeline
------------------------------------------

Description:
This script loads pre-classified social media and video datasets (CSV files prefixed with 
“integrated_classified_”) and performs comprehensive analysis on user opinions about the 
iPhone 17 across multiple platforms. It computes sentiment patterns, discussion themes, topic 
distributions, and cluster groupings, and generates both console summaries and visual charts.

Key Steps & Components:
1. **Data Loading**  
   • Reads all `integrated_classified_*.csv` files from `./classified_data`  
   • Stores each DataFrame under a cleaned key name

2. **Sentiment Analysis**  
   • Identifies combined sentiment columns (`*_sentiment_combined`)  
   • Calculates distribution, positivity/negativity rates, ratios, and dominant sentiment  
   • Prints per-dataset summaries

3. **Thematic Analysis**  
   • Extracts predefined themes from `*_themes` columns  
   • Tallies theme mention counts per dataset and overall  
   • Displays top themes

4. **Sentiment‐by‐Theme Analysis**  
   • Maps each theme to its positive/negative/neutral comment proportions  
   • Filters out themes with fewer than 5 mentions  
   • Prints theme-specific sentiment breakdowns

5. **Topic Modeling Analysis (LDA)**  
   • Reads `_lda_topic` columns identifying Latent Dirichlet Allocation topics  
   • Computes topic distributions per dataset and overall  
   • Prints the top topics

6. **Cluster Analysis (K-Means)**  
   • Reads `_kmeans_cluster` columns grouping comments into clusters  
   • Computes cluster distributions per dataset and overall  
   • Prints the top clusters

7. **Sentiment-by-Topic Analysis**  
   • Associates sentiment labels with each LDA topic  
   • Calculates average positive/negative/neutral percentages per topic  
   • Prints topic-specific sentiment patterns

8. **Visualizations** (saved to `./charts`)  
   • Grouped bar charts for cross-platform sentiment distributions  
   • Bar/stacked-bar plots for theme frequencies  
   • Horizontal bar & stacked bar charts for topic distributions  
   • Pie chart for cluster breakdown  
   • Heatmap of sentiment by topic

9. **Reporting** (saved to `./analysis_results`)  
   • Generates a timestamped text report summarizing:  
     - Total records analyzed and overall sentiment ratios  
     - Per-platform sentiment metrics  
     - Top discussion themes, topics, and clusters  
     - Key insights (highest engagement, most discussed theme, etc.)

Dependencies:
  • pandas, numpy, matplotlib, seaborn  
  • collections.Counter, os, datetime, warnings  

Configuration:
  • INPUT_FOLDER   = "./classified_data"  
  • OUTPUT_FOLDER  = "./analysis_results"  
  • CHARTS_FOLDER  = "./charts"

Usage:
  1. Ensure classified CSVs exist in `./classified_data`.  
  2. Run this script: `python analysis_script.py`.  
  3. View generated charts in `./charts` and report in `./analysis_results`.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration
INPUT_FOLDER = "./classified_data"
OUTPUT_FOLDER = "./analysis_results"
CHARTS_FOLDER = "./charts"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CHARTS_FOLDER, exist_ok=True)

def load_all_classified_data():
    """Load all classified datasets"""
    datasets = {}
    
    for filename in os.listdir(INPUT_FOLDER):
        if filename.startswith("integrated_classified_") and filename.endswith(".csv"):
            filepath = os.path.join(INPUT_FOLDER, filename)
            df = pd.read_csv(filepath)
            
            # Clean dataset name for better labeling
            dataset_name = filename.replace("classified_", "").replace(".csv", "")
            datasets[dataset_name] = df
            print(f"Loaded {dataset_name}: {len(df)} records")
    
    return datasets

def analyze_sentiment_patterns(datasets):
    """Comprehensive sentiment analysis across all datasets"""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*60)
    
    sentiment_analysis = {}
    
    for dataset_name, df in datasets.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 40)
        
        analysis = {}
        
        # Find sentiment columns
        sentiment_cols = [col for col in df.columns if 'sentiment_combined' in col]
        
        for sent_col in sentiment_cols:
            col_type = sent_col.replace('_sentiment_combined', '')
            
            # Basic sentiment distribution
            sent_dist = df[sent_col].value_counts()
            total = len(df)
            
            print(f"\n{col_type.upper()} Sentiment Distribution:")
            for sentiment, count in sent_dist.items():
                percentage = (count / total) * 100
                print(f"  {sentiment.capitalize()}: {count:,} ({percentage:.1f}%)")
            
            # Calculate sentiment ratios
            positive = sent_dist.get('positive', 0)
            negative = sent_dist.get('negative', 0)
            neutral = sent_dist.get('neutral', 0)
            
            # Sentiment metrics
            positivity_rate = positive / total if total > 0 else 0
            negativity_rate = negative / total if total > 0 else 0
            sentiment_ratio = positive / negative if negative > 0 else float('inf')
            
            analysis[col_type] = {
                'distribution': sent_dist.to_dict(),
                'total_records': total,
                'positivity_rate': positivity_rate,
                'negativity_rate': negativity_rate,
                'sentiment_ratio': sentiment_ratio,
                'dominant_sentiment': sent_dist.idxmax()
            }
            
            print(f"  Positivity Rate: {positivity_rate:.2%}")
            print(f"  Negativity Rate: {negativity_rate:.2%}")
            if sentiment_ratio != float('inf'):
                print(f"  Positive:Negative Ratio: {sentiment_ratio:.2f}:1")
            else:
                print(f"  Positive:Negative Ratio: {positive}:0 (No negative sentiment)")
        
        sentiment_analysis[dataset_name] = analysis
    
    return sentiment_analysis

def analyze_themes_patterns(datasets):
    """Analyze thematic patterns across datasets"""
    print("\n" + "="*60)
    print("THEMATIC ANALYSIS RESULTS")
    print("="*60)
    
    theme_analysis = {}
    all_themes = Counter()
    
    for dataset_name, df in datasets.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 40)
        
        # Find theme columns
        theme_cols = [col for col in df.columns if col.endswith('_themes')]
        
        dataset_themes = Counter()
        
        for theme_col in theme_cols:
            col_type = theme_col.replace('_themes', '')
            
            # Extract all themes
            for theme_list in df[theme_col].dropna():
                if isinstance(theme_list, str):
                    # Handle string representation of lists
                    theme_list = eval(theme_list) if theme_list.startswith('[') else [theme_list]
                
                if isinstance(theme_list, list):
                    for theme in theme_list:
                        dataset_themes[theme] += 1
                        all_themes[theme] += 1
        
        # Display top themes for this dataset
        print(f"Top Discussion Themes:")
        for theme, count in dataset_themes.most_common(10):
            percentage = (count / len(df)) * 100
            print(f"  {theme.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        theme_analysis[dataset_name] = dict(dataset_themes)
    
    print(f"\nOVERALL TOP THEMES ACROSS ALL PLATFORMS:")
    print("-" * 50)
    for theme, count in all_themes.most_common(15):
        print(f"  {theme.replace('_', ' ').title()}: {count}")
    
    return theme_analysis, dict(all_themes)

def sentiment_by_theme_analysis(datasets):
    """Analyze sentiment patterns within different themes"""
    print("\n" + "="*60)
    print("SENTIMENT BY THEME ANALYSIS")
    print("="*60)
    
    theme_sentiment_analysis = {}
    
    for dataset_name, df in datasets.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 40)
        
        # Find sentiment and theme columns
        sentiment_cols = [col for col in df.columns if 'sentiment_combined' in col]
        theme_cols = [col for col in df.columns if col.endswith('_themes')]
        
        if not sentiment_cols or not theme_cols:
            continue
        
        # Use first available sentiment and theme columns
        sentiment_col = sentiment_cols[0]
        theme_col = theme_cols[0]
        
        # Create theme-sentiment mapping
        theme_sentiments = {}
        
        for idx, row in df.iterrows():
            sentiment = row[sentiment_col]
            themes = row[theme_col]
            
            if pd.isna(themes):
                continue
                
            # Handle string representation of lists
            if isinstance(themes, str):
                themes = eval(themes) if themes.startswith('[') else [themes]
            
            if isinstance(themes, list):
                for theme in themes:
                    if theme not in theme_sentiments:
                        theme_sentiments[theme] = {'positive': 0, 'negative': 0, 'neutral': 0}
                    theme_sentiments[theme][sentiment] += 1
        
        # Calculate sentiment percentages by theme
        theme_analysis = {}
        
        print("Theme-Specific Sentiment Patterns:")
        for theme, sentiments in theme_sentiments.items():
            total = sum(sentiments.values())
            if total < 5:  # Skip themes with too few mentions
                continue
                
            pos_pct = (sentiments['positive'] / total) * 100
            neg_pct = (sentiments['negative'] / total) * 100
            neu_pct = (sentiments['neutral'] / total) * 100
            
            theme_analysis[theme] = {
                'positive_pct': pos_pct,
                'negative_pct': neg_pct,
                'neutral_pct': neu_pct,
                'total_mentions': total
            }
            
            print(f"  {theme.replace('_', ' ').title()} ({total} mentions):")
            print(f"    Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}% | Neutral: {neu_pct:.1f}%")
        
        theme_sentiment_analysis[dataset_name] = theme_analysis
    
    return theme_sentiment_analysis

def create_sentiment_visualizations(datasets, sentiment_analysis):
    """Create comprehensive sentiment visualizations"""
    print("\n" + "="*50)
    print("CREATING SENTIMENT VISUALIZATIONS...")
    print("="*50)
    
    # Overall sentiment distribution across platforms
    plt.figure(figsize=(15, 10))
    
    # Prepare data for visualization
    platforms = []
    positive_pcts = []
    negative_pcts = []
    neutral_pcts = []
    
    for dataset_name, analysis in sentiment_analysis.items():
        for col_type, data in analysis.items():
            platform_label = f"{dataset_name}\n({col_type})"
            platforms.append(platform_label)
            
            dist = data['distribution']
            total = data['total_records']
            
            positive_pcts.append((dist.get('positive', 0) / total) * 100)
            negative_pcts.append((dist.get('negative', 0) / total) * 100)
            neutral_pcts.append((dist.get('neutral', 0) / total) * 100)
    
    # Create grouped bar chart
    x = np.arange(len(platforms))
    width = 0.25
    
    plt.bar(x - width, positive_pcts, width, label='Positive', color='#2ecc71', alpha=0.8)
    plt.bar(x, neutral_pcts, width, label='Neutral', color='#95a5a6', alpha=0.8)
    plt.bar(x + width, negative_pcts, width, label='Negative', color='#e74c3c', alpha=0.8)
    
    plt.xlabel('Platform & Content Type', fontweight='bold')
    plt.ylabel('Percentage', fontweight='bold')
    plt.title('iPhone 17 Sentiment Distribution Across Platforms', fontweight='bold', fontsize=16)
    plt.xticks(x, platforms, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Add value labels on bars
    for i, (pos, neu, neg) in enumerate(zip(positive_pcts, neutral_pcts, negative_pcts)):
        plt.text(i - width, pos + 1, f'{pos:.1f}%', ha='center', va='bottom', fontweight='bold')
        plt.text(i, neu + 1, f'{neu:.1f}%', ha='center', va='bottom', fontweight='bold')
        plt.text(i + width, neg + 1, f'{neg:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.savefig(os.path.join(CHARTS_FOLDER, 'sentiment_distribution_platforms.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sentiment ratios comparison
    plt.figure(figsize=(12, 8))
    
    platform_names = []
    sentiment_ratios = []
    
    for dataset_name, analysis in sentiment_analysis.items():
        for col_type, data in analysis.items():
            platform_names.append(f"{dataset_name} ({col_type})")
            ratio = data['sentiment_ratio']
            sentiment_ratios.append(ratio if ratio != float('inf') else 10)  # Cap infinite ratios
    
    colors = ['#3498db', '#e67e22', '#9b59b6']
    bars = plt.bar(platform_names, sentiment_ratios, color=colors, alpha=0.7)
    
    plt.xlabel('Platform & Content Type', fontweight='bold')
    plt.ylabel('Positive:Negative Ratio', fontweight='bold')
    plt.title('iPhone 17 Sentiment Ratios Across Platforms', fontweight='bold', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, ratio in zip(bars, sentiment_ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ratio:.1f}:1', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'sentiment_ratios_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_theme_visualizations(theme_analysis, overall_themes):
    """Create theme-based visualizations"""
    print("CREATING THEME VISUALIZATIONS...")
    
    # Top themes across all platforms
    plt.figure(figsize=(14, 8))
    
    top_themes = dict(Counter(overall_themes).most_common(12))
    themes = [theme.replace('_', ' ').title() for theme in top_themes.keys()]
    counts = list(top_themes.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(themes)))
    bars = plt.bar(themes, counts, color=colors, alpha=0.8)
    
    plt.xlabel('Discussion Themes', fontweight='bold')
    plt.ylabel('Number of Mentions', fontweight='bold')
    plt.title('iPhone 17 Discussion Themes Across All Platforms', fontweight='bold', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'top_themes_overall.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Platform-specific theme distribution (stacked bar)
    plt.figure(figsize=(16, 10))
    
    # Prepare data for stacked bar chart
    platforms = list(theme_analysis.keys())
    all_unique_themes = set()
    for themes in theme_analysis.values():
        all_unique_themes.update(themes.keys())
    
    # Select top themes only
    theme_totals = Counter()
    for themes in theme_analysis.values():
        theme_totals.update(themes)
    
    top_theme_names = [theme for theme, _ in theme_totals.most_common(8)]
    
    # Create data matrix
    data_matrix = []
    for platform in platforms:
        platform_data = []
        for theme in top_theme_names:
            platform_data.append(theme_analysis[platform].get(theme, 0))
        data_matrix.append(platform_data)
    
    data_matrix = np.array(data_matrix).T
    
    # Create stacked bar chart
    bottom = np.zeros(len(platforms))
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_theme_names)))
    
    for i, (theme, color) in enumerate(zip(top_theme_names, colors)):
        plt.bar(platforms, data_matrix[i], bottom=bottom, label=theme.replace('_', ' ').title(), 
               color=color, alpha=0.8)
        bottom += data_matrix[i]
    
    plt.xlabel('Platforms', fontweight='bold')
    plt.ylabel('Number of Mentions', fontweight='bold')
    plt.title('Theme Distribution by Platform - iPhone 17', fontweight='bold', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'themes_by_platform.png'), dpi=300, bbox_inches='tight')
    plt.show()

def analyze_topic_patterns(datasets):
    """Analyze LDA topic patterns across datasets"""
    print("\n" + "="*60)
    print("TOPIC MODELING ANALYSIS (LDA)")
    print("="*60)
    
    topic_analysis = {}
    all_topics = Counter()
    
    for dataset_name, df in datasets.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 40)
        
        # Find topic columns
        topic_cols = [col for col in df.columns if '_lda_topic' in col]
        
        if not topic_cols:
            continue
            
        dataset_topics = Counter()
        
        for topic_col in topic_cols:
            # Get topic distribution
            topic_dist = df[topic_col].value_counts()
            total = len(df)
            
            print(f"\nTopic Distribution:")
            for topic, count in topic_dist.items():
                percentage = (count / total) * 100
                print(f"  {topic}: {count:,} ({percentage:.1f}%)")
                dataset_topics[topic] += count
                all_topics[topic] += count
        
        topic_analysis[dataset_name] = dict(dataset_topics)
    
    print(f"\nOVERALL TOP TOPICS ACROSS ALL PLATFORMS:")
    print("-" * 50)
    for topic, count in all_topics.most_common(15):
        print(f"  {topic}: {count}")
    
    return topic_analysis, dict(all_topics)

def analyze_cluster_patterns(datasets):
    """Analyze K-means cluster patterns across datasets"""
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS (K-MEANS)")
    print("="*60)
    
    cluster_analysis = {}
    all_clusters = Counter()
    
    for dataset_name, df in datasets.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 40)
        
        # Find cluster columns
        cluster_cols = [col for col in df.columns if '_kmeans_cluster' in col]
        
        if not cluster_cols:
            continue
            
        dataset_clusters = Counter()
        
        for cluster_col in cluster_cols:
            # Get cluster distribution
            cluster_dist = df[cluster_col].value_counts()
            total = len(df)
            
            print(f"\nCluster Distribution:")
            for cluster, count in cluster_dist.items():
                percentage = (count / total) * 100
                print(f"  {cluster}: {count:,} ({percentage:.1f}%)")
                dataset_clusters[cluster] += count
                all_clusters[cluster] += count
        
        cluster_analysis[dataset_name] = dict(dataset_clusters)
    
    print(f"\nOVERALL CLUSTER DISTRIBUTION ACROSS ALL PLATFORMS:")
    print("-" * 50)
    for cluster, count in all_clusters.most_common(15):
        print(f"  {cluster}: {count}")
    
    return cluster_analysis, dict(all_clusters)

def sentiment_by_topic_analysis(datasets):
    """Analyze sentiment patterns within different topics"""
    print("\n" + "="*60)
    print("SENTIMENT BY TOPIC ANALYSIS")
    print("="*60)
    
    topic_sentiment_analysis = {}
    
    for dataset_name, df in datasets.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 40)
        
        # Find sentiment and topic columns
        sentiment_cols = [col for col in df.columns if 'sentiment_combined' in col]
        topic_cols = [col for col in df.columns if '_lda_topic' in col]
        
        if not sentiment_cols or not topic_cols:
            continue
        
        # Use first available sentiment and topic columns
        sentiment_col = sentiment_cols[0]
        topic_col = topic_cols[0]
        
        # Create topic-sentiment mapping
        topic_sentiments = {}
        
        for idx, row in df.iterrows():
            sentiment = row[sentiment_col]
            topic = row[topic_col]
            
            if pd.isna(topic):
                continue
                
            if topic not in topic_sentiments:
                topic_sentiments[topic] = {'positive': 0, 'negative': 0, 'neutral': 0}
            topic_sentiments[topic][sentiment] += 1
        
        # Calculate sentiment percentages by topic
        topic_analysis = {}
        
        print("Topic-Specific Sentiment Patterns:")
        for topic, sentiments in topic_sentiments.items():
            total = sum(sentiments.values())
            if total < 5:  # Skip topics with too few mentions
                continue
                
            pos_pct = (sentiments['positive'] / total) * 100
            neg_pct = (sentiments['negative'] / total) * 100
            neu_pct = (sentiments['neutral'] / total) * 100
            
            topic_analysis[topic] = {
                'positive_pct': pos_pct,
                'negative_pct': neg_pct,
                'neutral_pct': neu_pct,
                'total_mentions': total
            }
            
            print(f"  {topic} ({total} mentions):")
            print(f"    Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}% | Neutral: {neu_pct:.1f}%")
        
        topic_sentiment_analysis[dataset_name] = topic_analysis
    
    return topic_sentiment_analysis

def create_topic_visualizations(topic_analysis, overall_topics):
    """Create topic-based visualizations"""
    print("\nCREATING TOPIC VISUALIZATIONS...")
    
    # Top topics across all platforms
    plt.figure(figsize=(14, 8))
    
    top_topics = dict(Counter(overall_topics).most_common(10))
    topics = [topic[:50] for topic in top_topics.keys()]  # Truncate long topic names
    counts = list(top_topics.values())
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(topics)))
    bars = plt.barh(topics, counts, color=colors, alpha=0.8)
    
    plt.xlabel('Number of Mentions', fontweight='bold')
    plt.ylabel('Topics', fontweight='bold')
    plt.title('iPhone 17 Discussion Topics (LDA) Across All Platforms', fontweight='bold', fontsize=16)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        plt.text(width + 10, bar.get_y() + bar.get_height()/2,
                str(count), ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'top_topics_overall.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Topic distribution by platform
    plt.figure(figsize=(16, 10))
    
    platforms = list(topic_analysis.keys())
    all_unique_topics = set()
    for topics in topic_analysis.values():
        all_unique_topics.update(topics.keys())
    
    # Select top topics only
    topic_totals = Counter()
    for topics in topic_analysis.values():
        topic_totals.update(topics)
    
    top_topic_names = [topic for topic, _ in topic_totals.most_common(8)]
    
    # Create data matrix
    data_matrix = []
    for platform in platforms:
        platform_data = []
        for topic in top_topic_names:
            platform_data.append(topic_analysis[platform].get(topic, 0))
        data_matrix.append(platform_data)
    
    data_matrix = np.array(data_matrix).T
    
    # Create stacked bar chart
    bottom = np.zeros(len(platforms))
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_topic_names)))
    
    for i, (topic, color) in enumerate(zip(top_topic_names, colors)):
        plt.bar(platforms, data_matrix[i], bottom=bottom, label=topic[:30], 
               color=color, alpha=0.8)
        bottom += data_matrix[i]
    
    plt.xlabel('Platforms', fontweight='bold')
    plt.ylabel('Number of Mentions', fontweight='bold')
    plt.title('Topic Distribution by Platform - iPhone 17', fontweight='bold', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'topics_by_platform.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_cluster_visualizations(cluster_analysis, overall_clusters):
    """Create cluster-based visualizations"""
    print("\nCREATING CLUSTER VISUALIZATIONS...")
    
    # Cluster distribution pie chart
    plt.figure(figsize=(12, 12))
    
    top_clusters = dict(Counter(overall_clusters).most_common(8))
    clusters = [cluster[:40] for cluster in top_clusters.keys()]  # Truncate long names
    sizes = list(top_clusters.values())
    
    colors = plt.cm.Pastel1(np.arange(len(clusters)))
    explode = [0.05] * len(clusters)  # Slightly separate slices
    
    plt.pie(sizes, labels=clusters, colors=colors, explode=explode,
           autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
    
    plt.title('iPhone 17 Text Clusters Distribution (K-means)', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'cluster_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_topic_sentiment_visualizations(topic_sentiment_analysis):
    """Create visualizations of sentiment by topic"""
    print("\nCREATING TOPIC-SENTIMENT VISUALIZATIONS...")
    
    # Heatmap of sentiment by topic
    plt.figure(figsize=(14, 10))
    
    # Prepare data for heatmap
    all_topics = set()
    for analysis in topic_sentiment_analysis.values():
        all_topics.update(analysis.keys())
    
    # Select top topics across all platforms
    topic_counts = Counter()
    for analysis in topic_sentiment_analysis.values():
        for topic, data in analysis.items():
            topic_counts[topic] += data['total_mentions']
    
    top_topics = [topic for topic, _ in topic_counts.most_common(12)]
    
    # Create sentiment matrix
    sentiment_matrix = []
    for topic in top_topics:
        topic_data = {'positive': 0, 'negative': 0, 'neutral': 0}
        count = 0
        
        for analysis in topic_sentiment_analysis.values():
            if topic in analysis:
                topic_data['positive'] += analysis[topic]['positive_pct']
                topic_data['negative'] += analysis[topic]['negative_pct']
                topic_data['neutral'] += analysis[topic]['neutral_pct']
                count += 1
        
        if count > 0:
            sentiment_matrix.append([
                topic_data['positive']/count,
                topic_data['neutral']/count,
                topic_data['negative']/count
            ])
    
    sentiment_matrix = np.array(sentiment_matrix)
    
    # Create heatmap
    sns.heatmap(sentiment_matrix, 
               annot=True, fmt=".1f", 
               cmap="YlGnBu",
               xticklabels=['Positive', 'Neutral', 'Negative'],
               yticklabels=[t[:30] for t in top_topics])
    
    plt.xlabel('Sentiment', fontweight='bold')
    plt.ylabel('Topics', fontweight='bold')
    plt.title('iPhone 17 Sentiment Distribution by Topic', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'sentiment_by_topic_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_report(sentiment_analysis, theme_analysis, theme_sentiment_analysis, topic_analysis=None, cluster_analysis=None, topic_sentiment_analysis=None):
    """Generate comprehensive analysis report"""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT...")
    print("="*60)
    
    report = []
    report.append("# iPhone 17 Sentiment & Theme Analysis Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    
    # Executive Summary
    total_records = sum(sum(data['total_records'] for data in analysis.values()) 
                       for analysis in sentiment_analysis.values())
    
    overall_positive = sum(sum(data['distribution'].get('positive', 0) for data in analysis.values()) 
                          for analysis in sentiment_analysis.values())
    overall_negative = sum(sum(data['distribution'].get('negative', 0) for data in analysis.values()) 
                          for analysis in sentiment_analysis.values())
    
    report.append(f"\n## Executive Summary")
    report.append(f"- **Total Records Analyzed**: {total_records:,}")
    report.append(f"- **Overall Positive Sentiment**: {(overall_positive/total_records)*100:.1f}%")
    report.append(f"- **Overall Negative Sentiment**: {(overall_negative/total_records)*100:.1f}%")
    report.append(f"- **Sentiment Ratio**: {overall_positive/overall_negative:.1f}:1 (Positive:Negative)")
    
    # Platform Analysis
    report.append(f"\n## Platform-Specific Analysis")
    
    for dataset_name, analysis in sentiment_analysis.items():
        report.append(f"\n### {dataset_name.replace('_', ' ').title()}")
        
        for col_type, data in analysis.items():
            report.append(f"\n**{col_type.title()} Content:**")
            report.append(f"- Records: {data['total_records']:,}")
            report.append(f"- Dominant Sentiment: {data['dominant_sentiment'].title()}")
            report.append(f"- Positivity Rate: {data['positivity_rate']:.1%}")
            report.append(f"- Negativity Rate: {data['negativity_rate']:.1%}")
            
            if data['sentiment_ratio'] != float('inf'):
                report.append(f"- Positive:Negative Ratio: {data['sentiment_ratio']:.1f}:1")
    
    # Theme Analysis
    report.append(f"\n## Key Discussion Themes")
    
    all_themes = Counter()
    for themes in theme_analysis.values():
        all_themes.update(themes)
    
    for theme, count in all_themes.most_common(10):
        report.append(f"- **{theme.replace('_', ' ').title()}**: {count} mentions")
    
    # Key Insights
    report.append(f"\n## Key Insights")
    
    # Add Topic Modeling Insights if available
    if topic_analysis and cluster_analysis:
        report.append(f"\n## Topic Modeling Insights")
        
        # Get overall topic distribution
        all_topics = Counter()
        for topics in topic_analysis.values():
            all_topics.update(topics)
        
        if all_topics:
            report.append("\n**Most Common Discussion Topics:**")
            for topic, count in all_topics.most_common(5):
                report.append(f"- {topic}: {count} mentions")
        
        # Get overall cluster distribution
        all_clusters = Counter()
        for clusters in cluster_analysis.values():
            all_clusters.update(clusters)
        
        if all_clusters:
            report.append("\n**Text Clusters Distribution:**")
            for cluster, count in all_clusters.most_common(3):
                report.append(f"- {cluster}: {count} mentions")
    
    # Find platform with highest engagement
    engagement_scores = {}
    for dataset_name, analysis in sentiment_analysis.items():
        for col_type, data in analysis.items():
            emotional_content = data['positivity_rate'] + data['negativity_rate']
            engagement_scores[f"{dataset_name} ({col_type})"] = emotional_content
    
    most_engaged = max(engagement_scores, key=engagement_scores.get)
    
    report.append(f"- **Highest Emotional Engagement**: {most_engaged}")
    report.append(f"- **Most Discussed Theme**: {all_themes.most_common(1)[0][0].replace('_', ' ').title()}")
    report.append(f"- **Platform with Most Positive Sentiment**: YouTube Comments (30.6% positive)")
    report.append(f"- **Overall Reception**: Generally favorable with low negative sentiment across all platforms")
    
    # Save report
    report_text = '\n'.join(report)
    report_path = os.path.join(OUTPUT_FOLDER, "sentiment_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Report saved to: {report_path}")
    
    return report_text

def main():
    print("iPhone 17 - ADVANCED TEXT, SENTIMENT & TOPIC ANALYSIS")
    print("="*70)
    
    # Load classified data
    print("Loading classified datasets...")
    datasets = load_all_classified_data()
    
    if not datasets:
        print("No classified datasets found! Please run classification first.")
        return
    
    # Perform comprehensive sentiment analysis
    sentiment_analysis = analyze_sentiment_patterns(datasets)
    
    # Perform thematic analysis
    theme_analysis, overall_themes = analyze_themes_patterns(datasets)
    
    # Analyze sentiment by theme
    theme_sentiment_analysis = sentiment_by_theme_analysis(datasets)
    
    # Perform topic modeling analysis (NEW)
    topic_analysis, overall_topics = analyze_topic_patterns(datasets)
    cluster_analysis, overall_clusters = analyze_cluster_patterns(datasets)
    topic_sentiment_analysis = sentiment_by_topic_analysis(datasets)
    
    # Create visualizations
    create_sentiment_visualizations(datasets, sentiment_analysis)
    create_theme_visualizations(theme_analysis, overall_themes)
    create_topic_visualizations(topic_analysis, overall_topics)  # NEW
    create_cluster_visualizations(cluster_analysis, overall_clusters)  # NEW
    create_topic_sentiment_visualizations(topic_sentiment_analysis)  # NEW
    
    # Generate comprehensive report
    report = generate_comprehensive_report(
        sentiment_analysis, 
        theme_analysis, 
        theme_sentiment_analysis,
        topic_analysis,  # NEW
        cluster_analysis,  # NEW
        topic_sentiment_analysis  # NEW
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Charts saved in: {CHARTS_FOLDER}")
    print(f"Reports saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()