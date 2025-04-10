import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from sklearn.metrics.pairwise import cosine_similarity

# Assuming VectorSearch class is saved in vector_search.py
try:
    from vector_search import VectorSearch
except ImportError:
    print("Error: Ensure VectorSearch class is defined in vector_search.py or adjust import.")
    # You might need to copy the VectorSearch class definition here if not importing
    exit()

# --- Configuration ---
EMBEDDINGS_PATH = 'job_embeddings.npy'
IDS_PATH = 'job_ids.npy'
PROCESSED_DATA_PATH = 'jobs_processed.csv' # Optional: Load for context in qualitative analysis

# Index configuration (ensure consistency with how VectorSearch is used)
INDEX_DESCRIPTION = 'IndexFlatL2' # Using exact search for evaluation clarity
INDEX_DIMENSION = 384 # Dimension for all-MiniLM-L6-v2

# Evaluation parameters
QUALITATIVE_SAMPLE_SIZE = 50 # Number of jobs to sample for qualitative check
NEIGHBORS_K = 5 # Number of neighbors to find (N=5)
RANDOM_PAIRS_SAMPLE_SIZE = 5000 # Number of random pairs for distribution analysis
QUALITATIVE_OUTPUT_FILE = 'qualitative_analysis_results.csv'

# --- Helper Function ---
def l2_to_cosine_similarity(l2_distance, embedding_dim=None):
    """
    Converts L2 distance to cosine similarity.
    Assumes embeddings are normalized (which sentence-transformers models usually are).
    Formula: similarity = 1 - (L2_distance^2 / 2)
    """
    similarity = 1 - (np.square(l2_distance) / 2)
    # Clamp values slightly outside [-1, 1] due to floating point errors
    return np.clip(similarity, -1.0, 1.0)

# --- 1. Load Data ---
print("--- Loading Data ---")
try:
    embeddings = np.load(EMBEDDINGS_PATH)
    # Ensure embeddings are float32 for Faiss
    if embeddings.dtype != np.float32:
        print("Converting embeddings to float32...")
        embeddings = embeddings.astype(np.float32)

    job_ids = np.load(IDS_PATH, allow_pickle=True).tolist() # Load as list
    print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")
    print(f"Loaded {len(job_ids)} job IDs.")

    if embeddings.shape[0] != len(job_ids):
        raise ValueError("Mismatch between number of embeddings and number of IDs.")
    if embeddings.shape[1] != INDEX_DIMENSION:
         raise ValueError(f"Loaded embeddings dimension ({embeddings.shape[1]}) does not match expected dimension ({INDEX_DIMENSION}).")

    # Optional: Load processed data for context
    df_processed = None
    if os.path.exists(PROCESSED_DATA_PATH):
        try:
            df_processed = pd.read_csv(PROCESSED_DATA_PATH, index_col='lid') # Use lid as index for easy lookup
            print(f"Loaded processed data from {PROCESSED_DATA_PATH} for context.")
        except Exception as e:
            print(f"Warning: Could not load processed data for context: {e}")
            df_processed = None
    else:
        print(f"Warning: Processed data file {PROCESSED_DATA_PATH} not found. Qualitative analysis will lack job title context.")


except FileNotFoundError as e:
    print(f"Error: Required file not found. {e}")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- 2. Build or Load Faiss Index ---
print("\n--- Initializing Vector Search ---")
vector_searcher = VectorSearch(dimension=INDEX_DIMENSION, index_description=INDEX_DESCRIPTION)

# Build the index (assuming it's not pre-built and saved)
# Check if training is needed (unlikely for IndexFlatL2, but good practice)
if not vector_searcher.index.is_trained:
    print("Index requires training (should not happen for IndexFlatL2). Training...")
    # If using an index like IVF, you might need a subset for training
    # For simplicity here, we'd train on all data if needed.
    vector_searcher.train(embeddings) # Pass all embeddings if training needed

print("Adding embeddings to the index...")
# Add data in batches if memory is a concern, otherwise add all at once
batch_size = 10000 # Adjust batch size based on memory
for i in range(0, len(embeddings), batch_size):
    emb_batch = embeddings[i:i+batch_size]
    id_batch = job_ids[i:i+batch_size]
    vector_searcher.add(emb_batch, id_batch)
    print(f"Added batch {i//batch_size + 1}...")

print(f"Index built successfully. Total vectors: {vector_searcher.ntotal}")


# --- 3. Qualitative Analysis ---
print("\n--- Performing Qualitative Analysis ---")
if QUALITATIVE_SAMPLE_SIZE > vector_searcher.ntotal:
     print(f"Warning: Sample size ({QUALITATIVE_SAMPLE_SIZE}) is larger than index size ({vector_searcher.ntotal}). Adjusting sample size.")
     QUALITATIVE_SAMPLE_SIZE = vector_searcher.ntotal

# Select random indices from the dataset
random_indices = random.sample(range(vector_searcher.ntotal), QUALITATIVE_SAMPLE_SIZE)
query_embeddings_sample = embeddings[random_indices]
query_ids_sample = [job_ids[i] for i in random_indices]

# Search for N+1 neighbors (to exclude self-match)
print(f"Searching for {NEIGHBORS_K + 1} neighbors for {QUALITATIVE_SAMPLE_SIZE} sample jobs...")
distances, neighbor_indices_internal = vector_searcher.index.search(query_embeddings_sample, NEIGHBORS_K + 1)

# Process results and save
qualitative_results = []
nearest_neighbor_similarities = [] # For distribution analysis

print("Processing qualitative results...")
for i, query_idx in enumerate(random_indices):
    query_id = job_ids[query_idx]
    query_title = df_processed.loc[query_id, 'jobTitle'] if df_processed is not None and query_id in df_processed.index else "N/A"

    result_row = {'Query_ID': query_id, 'Query_Title': query_title}

    # Get neighbors (excluding the first one, which is the query itself)
    neighbor_faiss_ids = neighbor_indices_internal[i][1:] # Skip index 0 (self)
    neighbor_distances = distances[i][1:]

    valid_neighbor_count = 0
    for j, neighbor_faiss_id in enumerate(neighbor_faiss_ids):
        if neighbor_faiss_id != -1 and neighbor_faiss_id < len(vector_searcher.id_map):
            neighbor_id = vector_searcher.id_map[neighbor_faiss_id]
            neighbor_distance = neighbor_distances[j]
            # Calculate cosine similarity from L2 distance
            neighbor_similarity = l2_to_cosine_similarity(neighbor_distance)

            neighbor_title = df_processed.loc[neighbor_id, 'jobTitle'] if df_processed is not None and neighbor_id in df_processed.index else "N/A"

            result_row[f'Neighbor_{j+1}_ID'] = neighbor_id
            result_row[f'Neighbor_{j+1}_Title'] = neighbor_title
            result_row[f'Neighbor_{j+1}_Similarity'] = round(neighbor_similarity, 4)

            # Store the similarity of the *closest* neighbor (excluding self) for distribution plot
            if j == 0:
                nearest_neighbor_similarities.append(neighbor_similarity)
            valid_neighbor_count += 1
        else:
            # Handle cases where fewer than K neighbors are found or index is invalid
             result_row[f'Neighbor_{j+1}_ID'] = None
             result_row[f'Neighbor_{j+1}_Title'] = None
             result_row[f'Neighbor_{j+1}_Similarity'] = None

    # Fill remaining columns if fewer than K valid neighbors found
    for j in range(valid_neighbor_count, NEIGHBORS_K):
         result_row[f'Neighbor_{j+1}_ID'] = None
         result_row[f'Neighbor_{j+1}_Title'] = None
         result_row[f'Neighbor_{j+1}_Similarity'] = None


    qualitative_results.append(result_row)

# Save to CSV
results_df = pd.DataFrame(qualitative_results)
try:
    results_df.to_csv(QUALITATIVE_OUTPUT_FILE, index=False)
    print(f"Qualitative analysis results saved to {QUALITATIVE_OUTPUT_FILE}")
except Exception as e:
    print(f"Error saving qualitative results to CSV: {e}")


# --- 4. Similarity Score Distribution Analysis ---
print("\n--- Performing Similarity Score Distribution Analysis ---")

# Calculate similarities for random pairs (proxy for non-duplicates)
print(f"Calculating similarities for {RANDOM_PAIRS_SAMPLE_SIZE} random pairs...")
random_pair_similarities = []
num_vectors = embeddings.shape[0]
for _ in range(RANDOM_PAIRS_SAMPLE_SIZE):
    idx1, idx2 = random.sample(range(num_vectors), 2) # Ensure distinct indices
    # Calculate cosine similarity directly for random pairs
    sim = cosine_similarity(embeddings[idx1].reshape(1, -1), embeddings[idx2].reshape(1, -1))[0, 0]
    random_pair_similarities.append(sim)

print("Similarity calculations complete.")

# Create DataFrame for plotting
plot_data_nn = pd.DataFrame({'Similarity': nearest_neighbor_similarities, 'Type': 'Nearest Neighbor (Likely Duplicate)'})
plot_data_random = pd.DataFrame({'Similarity': random_pair_similarities, 'Type': 'Random Pair (Non-Duplicate)'})
plot_data = pd.concat([plot_data_nn, plot_data_random], ignore_index=True)

# Plot distributions
print("Plotting similarity distributions...")
plt.figure(figsize=(12, 6))
sns.histplot(data=plot_data, x='Similarity', hue='Type', bins=50, kde=True, common_norm=False, stat='density')
plt.title('Distribution of Cosine Similarity Scores')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
try:
    plt.savefig('similarity_distribution.png')
    print("Similarity distribution plot saved to similarity_distribution.png")
except Exception as e:
    print(f"Error saving plot: {e}")
plt.show()

print("\n--- Evaluation Script Finished ---")