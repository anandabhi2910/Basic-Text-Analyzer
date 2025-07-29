import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Configuration ---
# Assuming you moved the file directly into data/raw/
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'IMDB Dataset.csv')
print(f"Raw data path: {RAW_DATA_PATH}")
PROCESSED_DATA_PATH = os.path.join('..', 'data', 'processed', 'IMDB_processed.csv')


# Load the Data
def load_data(file_path):

    #Loads data from a CSV file into a pandas DataFrame

    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None
    
# Data cleaning
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    
def tokenize_text(text):
    #Tokenize text into words
    return word_tokenize(text)

def remove_stopwords(words):
    """Removes common English stopwords from a list of words."""
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]

def lemmatize_text(words):
    """Lemmatizes words to their base form."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def get_vader_sentiment(text):
    """Calculates VADER sentiment scores for a given text."""
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


# --- Preprocessing Pipeline ---
def preprocess_pipeline(df, text_column='review'):
    """Applies a full preprocessing pipeline to the specified text column."""
    if df is None:
        return None

    print("Starting text preprocessing pipeline...")
    # Apply cleaning
    df['cleaned_review'] = df[text_column].apply(clean_text)
    # Apply tokenization
    df['tokens'] = df['cleaned_review'].apply(tokenize_text)
    # Apply stop word removal
    df['tokens_no_stopwords'] = df['tokens'].apply(remove_stopwords)
    # Apply lemmatization
    df['lemmas'] = df['tokens_no_stopwords'].apply(lemmatize_text)

    # Convert list of lemmas back to string for easier storage/analysis if needed
    df['processed_text'] = df['lemmas'].apply(lambda x: ' '.join(x))
    # Apply VADER sentiment analysis to the cleaned text
    df['vader_scores'] = df['cleaned_review'].apply(get_vader_sentiment)
    # Extract compound score for simplicity
    df['vader_compound_score'] = df['vader_scores'].apply(lambda x: x['compound'])

    # Classify sentiment based on compound score
    # A common threshold: >0.05 is positive, <-0.05 is negative, else neutral
    df['vader_sentiment'] = df['vader_compound_score'].apply(
        lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')
    )

    print("Preprocessing complete.")
    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load the data
    df = load_data(RAW_DATA_PATH)

    if df is not None:
        # Preprocess the data
        processed_df = preprocess_pipeline(df.copy()) # Use a copy to avoid modifying original df

        if processed_df is not None:
            # Display sample of original and processed text
            print("\n--- Sample Original vs. Processed Text ---")
            print(processed_df[['review', 'processed_text']].head())

            # Display counts of tokens before/after stopword removal and lemmatization
            print("\n--- Token Counts Sample ---")
            processed_df['original_token_count'] = processed_df['tokens'].apply(len)
            processed_df['processed_token_count'] = processed_df['lemmas'].apply(len)
            print(processed_df[['original_token_count', 'processed_token_count']].head())

            # Save the processed data
            # Ensure the processed directory exists
            os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
            processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
            print(f"\nProcessed data saved to: {PROCESSED_DATA_PATH}")
            print("Columns in processed DataFrame:", processed_df.columns.tolist())

