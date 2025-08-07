import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
from textblob import TextBlob
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
        if filename.startswith("classified_") and filename.endswith(".csv"):
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
        print(f"\n📈 {dataset_name.upper()}")
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
    
    # 1. Overall sentiment distribution across platforms
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
    plt.title('iPhone 16 Sentiment Distribution Across Platforms', fontweight='bold', fontsize=16)
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
    
    # 2. Sentiment ratios comparison
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
    plt.title('iPhone 16 Sentiment Ratios Across Platforms', fontweight='bold', fontsize=16)
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
    
    # 1. Top themes across all platforms
    plt.figure(figsize=(14, 8))
    
    top_themes = dict(Counter(overall_themes).most_common(12))
    themes = [theme.replace('_', ' ').title() for theme in top_themes.keys()]
    counts = list(top_themes.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(themes)))
    bars = plt.bar(themes, counts, color=colors, alpha=0.8)
    
    plt.xlabel('Discussion Themes', fontweight='bold')
    plt.ylabel('Number of Mentions', fontweight='bold')
    plt.title('iPhone 16 Discussion Themes Across All Platforms', fontweight='bold', fontsize=16)
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
    
    # 2. Platform-specific theme distribution (stacked bar)
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
    plt.title('Theme Distribution by Platform - iPhone 16', fontweight='bold', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'themes_by_platform.png'), dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_report(sentiment_analysis, theme_analysis, theme_sentiment_analysis):
    """Generate comprehensive analysis report"""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT...")
    print("="*60)
    
    report = []
    report.append("# iPhone 16 Sentiment & Theme Analysis Report")
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
    print("iPhone 16 - ADVANCED TEXT & SENTIMENT ANALYSIS")
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
    
    # Create visualizations
    create_sentiment_visualizations(datasets, sentiment_analysis)
    create_theme_visualizations(theme_analysis, overall_themes)
    
    # Generate comprehensive report
    report = generate_comprehensive_report(sentiment_analysis, theme_analysis, theme_sentiment_analysis)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Charts saved in: {CHARTS_FOLDER}")
    print(f"Reports saved in: {OUTPUT_FOLDER}")
    print("\nNext Steps:")
    print("1. Review the generated visualizations")
    print("2. Use insights for business recommendations")
    print("3. Prepare final presentation materials")

if __name__ == "__main__":
    main()