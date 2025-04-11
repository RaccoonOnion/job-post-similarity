import pandas as pd
from bs4 import BeautifulSoup
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(csv_file_path):
    """Loads the initial CSV data."""
    logging.info(f"Attempting to load data from: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
        logging.info("Data loaded successfully.")
        logging.info(f"Original DataFrame shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: The file {csv_file_path} was not found.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during data loading: {e}")
        raise

def clean_html(df, raw_col='jobDescRaw', clean_col='jobDescClean'):
    """Extracts text from HTML content in a column."""
    if raw_col in df.columns:
        logging.info(f"Extracting text from HTML column '{raw_col}' into '{clean_col}'...")
        # Ensure raw column is string type and fill NaNs
        df[raw_col] = df[raw_col].astype(str).fillna('')
        df[clean_col] = df[raw_col].apply(lambda x: BeautifulSoup(x, "html.parser").get_text(separator=" "))
        logging.info(f"Created '{clean_col}' column.")
    else:
        logging.warning(f"Column '{raw_col}' not found. Skipping HTML cleaning.")
    return df


def handle_missing_values(df):
    """Handles missing values based on EDA findings."""
    logging.info("--- Handling Missing Values ---")
    # Fill categorical/location NaNs with 'Unknown' [cite: 37]
    fill_unknown_cols = ['companyName', 'finalZipcode', 'finalState', 'finalCity', 'companyBranchName']
    for col in fill_unknown_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            logging.info(f"Filled NaN in '{col}' with 'Unknown'.")

    # Drop rows where 'correctDate' is missing [cite: 37]
    if 'correctDate' in df.columns:
        initial_rows = df.shape[0]
        df.dropna(subset=['correctDate'], inplace=True)
        rows_dropped = initial_rows - df.shape[0]
        logging.info(f"Dropped {rows_dropped} rows with missing 'correctDate'.")
    else:
         logging.warning("Column 'correctDate' not found. Skipping dropna.")

    logging.info("Missing value handling complete.")
    return df

def remove_duplicates(df):
    """Removes duplicate rows based on key columns."""
    logging.info("--- Removing Duplicates ---")
    # key_cols = ['jobTitle', 'companyName', 'finalZipcode', 'correctDate'] # our old version, however, I realize there could be same role same day from same company but has different seniority. Use jobDescClean instead. [cite: 38]
    key_cols = ['jobDescClean'] # this column has been added in previous step.
    if all(col in df.columns for col in key_cols):
        initial_rows = df.shape[0]
        df.drop_duplicates(subset=key_cols, keep='first', inplace=True)
        rows_dropped = initial_rows - df.shape[0]
        logging.info(f"Dropped {rows_dropped} duplicate rows based on {key_cols}.")
    else:
        logging.warning(f"Skipping duplicate removal based on {key_cols} as one or more columns are missing.")
    return df

def clean_location_data(df):
    """Standardizes location-related columns."""
    logging.info("--- Cleaning Location Data ---")
    # Standardize 'finalState' [cite: 39]
    if 'finalState' in df.columns:
        df['finalState'] = df['finalState'].astype(str).str.replace(r'\s*,\s*$', '', regex=True).str.strip()
        logging.info("Cleaned 'finalState' (removed trailing commas/whitespace).")

    # Handle 'remote' in 'finalZipcode' [cite: 39]
    if 'finalZipcode' in df.columns:
        df['finalZipcode'] = df['finalZipcode'].astype(str).str.replace('remote', 'REMOTE', case=False)
        logging.info("Cleaned 'finalZipcode' (standardized 'remote').")

    # Standardize 'finalCity' [cite: 39]
    if 'finalCity' in df.columns:
        df['finalCity'] = df['finalCity'].astype(str).str.title().str.strip()
        logging.info("Cleaned 'finalCity' (converted to title case).")
    return df

def clean_text_data(df, text_col='jobDescClean'):
    """Applies basic cleaning to the text description column."""
    logging.info("--- Cleaning Job Description Text ---")
    if text_col in df.columns:
        # Convert to lowercase [cite: 40]
        df[text_col] = df[text_col].str.lower()
        # Remove extra whitespace [cite: 40]
        df[text_col] = df[text_col].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
        logging.info(f"Cleaned '{text_col}' (lowercase, extra whitespace removed).")
    else:
        logging.warning(f"Text column '{text_col}' not found. Skipping text cleaning.")
    return df

def drop_unused_columns(df):
    """Drops columns not needed for the embedding/similarity task."""
    logging.info("--- Dropping Unused Columns ---")
    columns_to_drop = [
        'jobDescRaw', 'jobDescLength', 'jobDescUrl', 'companyBranchName',
        'scrapedLocation', 'nlpBenefits', 'nlpSkills', 'nlpSoftSkills',
        'nlpDegreeLevel', 'nlpEmployment', 'nlpSeniority'
    ] # [cite: 41]
    # Drop only columns that actually exist
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df.drop(columns=existing_columns_to_drop, inplace=True)
        logging.info(f"Dropped columns: {existing_columns_to_drop}")
    else:
        logging.info("No specified columns found for dropping.")
    return df

def preprocess_data(input_csv_path='jobs.csv', output_csv_path='jobs_processed.csv'):
    """Main function to run all preprocessing steps."""
    df = load_data(input_csv_path)
    df = clean_html(df) # Ensure clean text column exists before other steps that might use it
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = clean_location_data(df)
    df = clean_text_data(df) # Clean the extracted text
    df = drop_unused_columns(df) # Drop raw HTML and other unused cols

    # Final check
    logging.info("--- Preprocessing Complete ---")
    logging.info(f"Processed DataFrame shape: {df.shape}")
    logging.info("\nFinal DataFrame Info:")
    df.info(verbose=False) # Use verbose=False for brevity in logging
    logging.info("\nFirst row of processed data (transposed):")
    logging.info(f"\n{df.head(1).T}") # Transpose for better logging readability

    # Save the processed data
    try:
        df.to_csv(output_csv_path, index=False)
        logging.info(f"Processed data saved to {output_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save processed data: {e}")

    return df

# Example of how to run the preprocessing
if __name__ == "__main__":
    # Assuming 'jobs.csv' is in the same directory or a specified path
    # The output 'jobs_processed.csv' will also be saved there by default
    data_folder = "data"
    processed_df = preprocess_data(input_csv_path=f'{data_folder}/jobs.csv', output_csv_path=f'{data_folder}/jobs_processed_2.csv')
    # You can now use 'processed_df' for the next steps (embedding generation)