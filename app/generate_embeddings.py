import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data(csv_path):
    """Loads the processed data CSV file."""
    logging.info(f"Loading processed data from: {csv_path}")
    if not os.path.exists(csv_path):
        logging.error(f"Error: Processed data file not found at {csv_path}")
        raise FileNotFoundError(f"File not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading data from {csv_path}: {e}")
        raise

def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Loads the Sentence Transformer model."""
    logging.info(f"Loading Sentence Transformer model: {model_name}")
    try:
        # trust_remote_code=True might be needed for some models, but not usually for standard ones
        model = SentenceTransformer(model_name)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading Sentence Transformer model '{model_name}': {e}")
        raise

def generate_embeddings(model, texts):
    """Generates embeddings for a list of texts."""
    logging.info(f"Generating embeddings for {len(texts)} documents...")
    start_time = time.time()
    try:
        embeddings = model.encode(texts, show_progress_bar=True)
        end_time = time.time()
        logging.info("Embeddings generated successfully.")
        logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
        logging.info(f"Shape of embeddings array: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logging.error(f"Error during embedding generation: {e}")
        raise

def save_embeddings_and_ids(embeddings, ids, embeddings_path, ids_path):
    """Saves embeddings and corresponding IDs to .npy files."""
    logging.info(f"Saving embeddings to: {embeddings_path}")
    try:
        np.save(embeddings_path, embeddings)
        logging.info("Embeddings saved successfully.")
    except Exception as e:
        logging.error(f"Error saving embeddings to {embeddings_path}: {e}")
        raise

    logging.info(f"Saving corresponding job IDs to: {ids_path}")
    try:
        np.save(ids_path, np.array(ids, dtype=object)) # Save as object array for strings
        logging.info("Job IDs saved successfully.")
    except Exception as e:
        logging.error(f"Error saving Job IDs to {ids_path}: {e}")
        raise

def process_and_generate_embeddings(
    input_csv_path='jobs_processed.csv',
    text_column='jobDescClean',
    id_column='lid',
    model_name='all-MiniLM-L6-v2',
    embeddings_output_path='job_embeddings.npy',
    ids_output_path='job_ids.npy'):
    """Loads data, generates embeddings, and saves them."""

    try:
        # 1. Load Data
        df_processed = load_processed_data(input_csv_path)

        # Ensure required columns exist
        if text_column not in df_processed.columns:
            raise ValueError(f"Error: Text column '{text_column}' not found in {input_csv_path}.")
        if id_column not in df_processed.columns:
            raise ValueError(f"Error: ID column '{id_column}' not found in {input_csv_path}.")

        # Handle potential NaNs in text column
        df_processed[text_column] = df_processed[text_column].fillna('').astype(str)
        texts_to_embed = df_processed[text_column].tolist()
        job_ids = df_processed[id_column].tolist()

        # 2. Load Model
        model = load_embedding_model(model_name)

        # 3. Generate Embeddings
        embeddings = generate_embeddings(model, texts_to_embed)

        # 4. Save Results
        save_embeddings_and_ids(embeddings, job_ids, embeddings_output_path, ids_output_path)

        logging.info("\nEmbedding generation process complete.")
        return embeddings, job_ids

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"Embedding generation failed: {e}")
        return None, None # Indicate failure

# Example of how to run the modularized script
if __name__ == "__main__":
    # Define file paths (consider using environment variables or config files)
    DATA_DIR = 'data' # Assuming data is in a subdirectory
    INPUT_CSV = os.path.join(DATA_DIR, 'jobs_processed.csv')
    EMBEDDINGS_OUT = os.path.join(DATA_DIR, 'job_embeddings.npy')
    IDS_OUT = os.path.join(DATA_DIR, 'job_ids.npy')

    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Run the process
    process_and_generate_embeddings(
        input_csv_path=INPUT_CSV,
        embeddings_output_path=EMBEDDINGS_OUT,
        ids_output_path=IDS_OUT
    )