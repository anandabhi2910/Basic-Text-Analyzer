import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from nltk.util import ngrams # For generating n-grams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder # For finding bigrams

# --- Configuration ---
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__),'..', 'data', 'processed', 'IMDB_processed.csv')
PLOTS_OUTPUT_DIR = os.path.join('..', 'output', 'plots')

# Ensure output directory exists
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

# --- Data Loading ---
def load_processed_data(file_path):
    """Loads processed data from a CSV file."""
    print(f"Loading processed data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        # Ensure the 'lemmas' column is treated as a list of strings if loaded as string
        # It might load as string representation of list, so convert it back
        # Example: "[word1, word2]" -> ["word1", "word2"]
        if isinstance(df['lemmas'].iloc[0], str):
            df['lemmas'] = df['lemmas'].apply(eval) # eval safely converts string representation of list to actual list
        print(f"Processed data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Processed file not found at {file_path}. Please ensure data_loader.py has been run.")
        return None
    except Exception as e:
        print(f"An error occurred while loading processed data: {e}")
        return None

# --- EDA Functions ---
def plot_word_frequency(df, column='lemmas', top_n=20, title="Most Frequent Words", filename="word_frequency.png"):
    """Plots the frequency of the most common words."""
    all_words = [word for sublist in df[column] for word in sublist]
    word_counts = Counter(all_words)

    most_common_words = word_counts.most_common(top_n)
    words, counts = zip(*most_common_words)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(words), y=list(counts), palette='viridis')
    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, filename))
    print(f"Saved: {filename}")
    plt.close() # Close plot to prevent display issues in scripts

def plot_text_length_distribution(df, original_col='review', processed_col='processed_text', filename="text_length_distribution.png"):
    """Plots the distribution of original and processed text lengths."""
    df['original_length'] = df[original_col].apply(len)
    df['processed_length'] = df[processed_col].apply(len)

    plt.figure(figsize=(14, 7))
    sns.histplot(df['original_length'], color='skyblue', label='Original Text Length', kde=True, bins=50)
    sns.histplot(df['processed_length'], color='salmon', label='Processed Text Length', kde=True, bins=50)
    plt.title("Distribution of Text Lengths (Original vs. Processed)")
    plt.xlabel("Text Length (Characters)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, filename))
    print(f"Saved: {filename}")
    plt.close()

def get_ngrams(tokens_list, n=2, top_n=20):
    """Generates and counts most common n-grams."""
    all_ngrams = []
    for tokens in tokens_list:
        all_ngrams.extend(list(ngrams(tokens, n)))

    ngram_counts = Counter(all_ngrams)
    return ngram_counts.most_common(top_n)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load the processed data
    df = load_processed_data(PROCESSED_DATA_PATH)

    if df is not None:
        print("\n--- Performing EDA ---")

        # 1. Plot Word Frequency
        plot_word_frequency(df, column='lemmas', title="Most Frequent Lemmatized Words (Top 20)")

        # 2. Plot Text Length Distribution
        plot_text_length_distribution(df)

        # 3. Analyze N-grams (e.g., Bigrams and Trigrams)
        print("\n--- Most Common Bigrams (Top 10) ---")
        bigrams = get_ngrams(df['lemmas'], n=2, top_n=10)
        for bigram, count in bigrams:
            print(f"'{' '.join(bigram)}': {count}")

        print("\n--- Most Common Trigrams (Top 10) ---")
        trigrams = get_ngrams(df['lemmas'], n=3, top_n=10)
        for trigram, count in trigrams:
            print(f"'{' '.join(trigram)}': {count}")

        # 4. Basic Sentiment Distribution (if 'sentiment' column exists)
        if 'sentiment' in df.columns:
            print("\n--- Sentiment Distribution ---")
            sentiment_counts = df['sentiment'].value_counts()
            print(sentiment_counts)

            plt.figure(figsize=(7, 5))
            sns.countplot(x='sentiment', data=df, palette='pastel')
            plt.title("Sentiment Distribution")
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "sentiment_distribution.png"))
            print("Saved: sentiment_distribution.png")
            plt.close()
        else:
            print("\n'sentiment' column not found. Skipping sentiment distribution plot.")


        print("\nEDA complete. Check the 'output/plots' directory for generated graphs.")