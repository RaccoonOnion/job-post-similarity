import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# --- Configuration ---
PROCESSED_DATA_PATH = 'jobs_processed.csv' # Input file from preprocessing
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Recommended Sentence Transformer model
TEXT_COLUMN = 'jobDescClean' # Column containing the cleaned job descriptions
EMBEDDINGS_OUTPUT_PATH = 'job_embeddings.npy' # File to save the generated embeddings
ID_COLUMN = 'lid' # Column containing the unique job identifier

# --- Load Processed Data ---
print(f"Loading processed data from: {PROCESSED_DATA_PATH}")
try:
    df_processed = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Loaded {len(df_processed)} rows.")

    # Ensure the text column exists
    if TEXT_COLUMN not in df_processed.columns:
        raise ValueError(f"Error: Text column '{TEXT_COLUMN}' not found in the CSV.")
    # Ensure the ID column exists
    if ID_COLUMN not in df_processed.columns:
        raise ValueError(f"Error: ID column '{ID_COLUMN}' not found in the CSV.")

    # Handle potential NaN values in the text column just in case but we have made sure of this in the preprocessing step
    df_processed[TEXT_COLUMN] = df_processed[TEXT_COLUMN].fillna('').astype(str)

except FileNotFoundError:
    print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    exit()

# --- Load Embedding Model ---
print(f"Loading Sentence Transformer model: {EMBEDDING_MODEL_NAME}")
# This will download the model automatically if not cached
try:
    # Set trust_remote_code=True if required by the specific model version,
    # but be aware of the security implications. For standard models like
    # all-MiniLM-L6-v2, it's typically not needed.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}")
    exit()

# --- Generate Embeddings ---
print(f"Generating embeddings for column: '{TEXT_COLUMN}'...")

# Get the list of texts to embed
texts_to_embed = df_processed[TEXT_COLUMN].tolist()

# Generate embeddings
# Use show_progress_bar=True for visual feedback on long processes
# Consider adjusting batch_size based on available memory (GPU/CPU)
start_time = time.time()
embeddings = model.encode(texts_to_embed, show_progress_bar=True)
end_time = time.time()

print(f"Embeddings generated successfully for {len(embeddings)} documents.")
print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"Shape of embeddings array: {embeddings.shape}") # Should be (num_documents, embedding_dimension)

# --- Store Embeddings and Mapping ---
# It's crucial to store the embeddings alongside their corresponding IDs

# Get the job IDs in the same order as the embeddings
job_ids = df_processed[ID_COLUMN].tolist()

# Create a dictionary mapping ID to index (row number in embeddings array)
# This isn't strictly necessary if we save IDs and embeddings separately
# but can be useful.
# id_to_index = {job_id: i for i, job_id in enumerate(job_ids)}

# Save the embeddings array to a .npy file for efficient loading later

# Saving the embeddings as a NumPy (.npy) file allows for very quick retrieval 
# compared to regenerating them every time you run your code. 
# Loading a .npy file using np.load() is efficient
# Libraries like Faiss directly accept NumPy arrays as input 
# when adding vectors to an index (index.add(embeddings)). 
# Loading from .npy provides the data in the exact format needed.

print(f"Saving embeddings to: {EMBEDDINGS_OUTPUT_PATH}")
try:
    np.save(EMBEDDINGS_OUTPUT_PATH, embeddings)
    print("Embeddings saved successfully.")

    # Optional: Save the corresponding job IDs as well
    # This ensures you can always map the rows in the .npy file back to the correct job ID
    ids_output_path = 'job_ids.npy'
    print(f"Saving corresponding job IDs to: {ids_output_path}")
    np.save(ids_output_path, np.array(job_ids, dtype=object)) # Save as object array to handle string IDs
    print("Job IDs saved successfully.")

except Exception as e:
    print(f"Error saving embeddings or IDs: {e}")

print("\nEmbedding generation process complete.")