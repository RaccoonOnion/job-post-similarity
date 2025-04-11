import os
import logging
import time
import numpy as np
import pandas as pd
import random # Added for sampling
from sklearn.metrics.pairwise import cosine_similarity

# --- Import modularized functions/classes ---
# Assuming these files are in the same directory or accessible via PYTHONPATH
try:
    from preprocess_data import preprocess_data
    from generate_embeddings import (
        load_processed_data,
        load_embedding_model,
        generate_embeddings,
        save_embeddings_and_ids
    )
    from vector_search import VectorSearch # Use the version with GPU support
    from evaluation import l2_to_cosine_similarity
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure preprocess_data.py, generate_embeddings.py, vector_search.py, and evaluation.py are accessible.")
    exit()

# --- Configuration from Environment Variables ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File/Directory Paths (relative to the script location inside the container: /app)
# Reads from Env Vars or uses defaults
DATA_DIR = os.environ.get('DATA_DIR', 'data')
ANALYSIS_DIR = os.environ.get('ANALYSIS_DIR', 'analysis_files')

RAW_CSV_PATH = os.path.join(DATA_DIR, 'jobs.csv')
PROCESSED_CSV_PATH = os.path.join(DATA_DIR, 'jobs_processed.csv')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'job_embeddings.npy')
IDS_PATH = os.path.join(DATA_DIR, 'job_ids.npy')
INDEX_PATH = os.path.join(DATA_DIR, 'faiss_index.index')
ID_MAP_PATH = os.path.join(DATA_DIR, 'id_map.pkl')
FINAL_OUTPUT_CSV = os.path.join(ANALYSIS_DIR, 'similarity_results.csv')

# Model & Search Parameters from Env Vars (with defaults)
EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
TEXT_COLUMN = os.environ.get('TEXT_COLUMN', 'jobDescClean')
ID_COLUMN = os.environ.get('ID_COLUMN', 'lid')
INDEX_DIMENSION = int(os.environ.get('INDEX_DIMENSION', '384'))
INDEX_DESCRIPTION = os.environ.get('INDEX_DESCRIPTION', 'HNSW32') # Default to HNSW
NEIGHBORS_K = int(os.environ.get('NEIGHBORS_K', '2'))
SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', '0.90'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '10000'))

# Search Sample Size from Env Var
search_sample_size_str = os.environ.get('SEARCH_SAMPLE_SIZE', '') # Read as string
SEARCH_SAMPLE_SIZE = int(search_sample_size_str) if search_sample_size_str.isdigit() else None # Convert to int or None

# GPU Configuration from Env Var
USE_GPU_STR = os.environ.get('USE_GPU', 'False').lower()
USE_GPU = USE_GPU_STR == 'true'

logging.info("--- Configuration Loaded ---")
logging.info(f"INDEX_DESCRIPTION: {INDEX_DESCRIPTION}")
logging.info(f"SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}")
logging.info(f"SEARCH_SAMPLE_SIZE: {SEARCH_SAMPLE_SIZE if SEARCH_SAMPLE_SIZE is not None else 'Full'}")
logging.info(f"USE_GPU: {USE_GPU}")
logging.info(f"DATA_DIR: {DATA_DIR}")
logging.info(f"ANALYSIS_DIR: {ANALYSIS_DIR}")
# --- ---

# --- Helper to build Faiss index ---
# (Keep build_faiss_index function as before)
def build_faiss_index(vector_searcher, embeddings, ids, batch_size=BATCH_SIZE):
    """Builds the Faiss index by adding embeddings in batches."""
    logging.info("--- Building Faiss Index ---")
    if not vector_searcher.index.is_trained:
        logging.info("Index requires training. Calling train method...")
        try:
            vector_searcher.train(embeddings)
        except Exception as e:
            logging.error(f"Faiss index training failed: {e}")
            raise

    logging.info("Adding embeddings to the index...")
    num_added = 0
    try:
        for i in range(0, len(embeddings), batch_size):
            emb_batch = embeddings[i:i+batch_size]
            id_batch = ids[i:i+batch_size]
            if len(emb_batch) > 0:
                 vector_searcher.add(emb_batch, id_batch)
                 num_added += len(emb_batch)
                 logging.info(f"Added batch {i//batch_size + 1}... Total added: {num_added}")
        logging.info(f"Index built successfully. Total vectors: {vector_searcher.ntotal}")
        if vector_searcher.ntotal != len(ids):
             logging.warning(f"Potential mismatch: Index size {vector_searcher.ntotal} vs loaded IDs {len(ids)}")
    except Exception as e:
        logging.error(f"Failed during index building: {e}")
        raise

# --- Main Pipeline Function ---
def run_similarity_pipeline():
    """Runs the full pipeline from raw data to similarity results."""
    logging.info("--- Starting Job Similarity Pipeline ---")
    start_pipeline_time = time.time()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # --- 1. Preprocessing ---
    logging.info("--- Step 1: Data Preprocessing ---")
    if not os.path.exists(PROCESSED_CSV_PATH):
        logging.info(f"Processed data '{PROCESSED_CSV_PATH}' not found. Running preprocessing...")
        try:
            # Check for raw CSV existence before calling preprocess
            if not os.path.exists(RAW_CSV_PATH):
                 logging.error(f"Raw data file '{RAW_CSV_PATH}' not found. Cannot preprocess.")
                 return # Exit if raw data is missing
            preprocess_data(input_csv_path=RAW_CSV_PATH, output_csv_path=PROCESSED_CSV_PATH)
            logging.info("Preprocessing completed.")
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}")
            return
    else:
        logging.info(f"Found existing processed data '{PROCESSED_CSV_PATH}'. Skipping preprocessing.")


    # --- 2. Embedding Generation ---
    logging.info("--- Step 2: Embedding Generation ---")
    embeddings = None
    job_ids = None
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(IDS_PATH):
        logging.info(f"Found existing embeddings '{EMBEDDINGS_PATH}' and IDs '{IDS_PATH}'. Loading...")
        try:
            embeddings = np.load(EMBEDDINGS_PATH)
            job_ids = np.load(IDS_PATH, allow_pickle=True).tolist()
            logging.info(f"Loaded {len(job_ids)} embeddings/IDs.")
            if embeddings.shape[1] != INDEX_DIMENSION: raise ValueError(f"Embeddings dim mismatch")
            if embeddings.dtype != np.float32: embeddings = embeddings.astype(np.float32)
            if len(embeddings) != len(job_ids): raise ValueError("Embeddings/IDs count mismatch")
        except Exception as e:
            logging.error(f"Failed to load existing embeddings/IDs: {e}. Will regenerate.")
            embeddings = None; job_ids = None

    if embeddings is None or job_ids is None:
        logging.info("Generating embeddings...")
        try:
            # Ensure processed data exists before trying to load for embedding generation
            if not os.path.exists(PROCESSED_CSV_PATH):
                logging.error(f"Processed data file '{PROCESSED_CSV_PATH}' is needed for embedding generation but not found.")
                return # Exit if processed data is missing

            df_processed = load_processed_data(PROCESSED_CSV_PATH) # Use function from generate_embeddings
            if TEXT_COLUMN not in df_processed.columns or ID_COLUMN not in df_processed.columns:
                 raise ValueError("Required columns missing in processed data for embedding generation.")
            df_processed[TEXT_COLUMN] = df_processed[TEXT_COLUMN].fillna('').astype(str)
            texts_to_embed = df_processed[TEXT_COLUMN].tolist()
            job_ids_gen = df_processed[ID_COLUMN].tolist()
            model = load_embedding_model(EMBEDDING_MODEL_NAME)
            embeddings_gen = generate_embeddings(model, texts_to_embed)
            save_embeddings_and_ids(embeddings_gen, job_ids_gen, EMBEDDINGS_PATH, IDS_PATH)
            embeddings = embeddings_gen; job_ids = job_ids_gen
            logging.info("Embeddings generated and saved.")
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}"); return

    if embeddings is None or job_ids is None:
         logging.error("Embeddings/IDs missing. Exiting."); return


    # --- 3. Vector Search Indexing ---
    logging.info("--- Step 3: Vector Search Indexing ---")
    vector_searcher = VectorSearch(
        dimension=INDEX_DIMENSION,
        index_description=INDEX_DESCRIPTION,
        use_gpu=USE_GPU
    )

    if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
        logging.info(f"Found existing index '{INDEX_PATH}' and ID map '{ID_MAP_PATH}'. Loading...")
        try:
            vector_searcher.load(INDEX_PATH, ID_MAP_PATH)
            if vector_searcher.ntotal != len(embeddings):
                 logging.warning(f"Index/Embeddings count mismatch ({vector_searcher.ntotal}/{len(embeddings)}). Rebuilding.")
                 vector_searcher = VectorSearch(dimension=INDEX_DIMENSION, index_description=INDEX_DESCRIPTION, use_gpu=USE_GPU)
                 build_faiss_index(vector_searcher, embeddings, job_ids)
                 vector_searcher.save(INDEX_PATH, ID_MAP_PATH)
            else:
                 logging.info("Index loaded successfully.")
                 logging.info(f"Index is using {'GPU' if vector_searcher.use_gpu else 'CPU'}.")

        except Exception as e:
            logging.error(f"Failed to load existing index/map: {e}. Will rebuild.")
            vector_searcher = VectorSearch(dimension=INDEX_DIMENSION, index_description=INDEX_DESCRIPTION, use_gpu=USE_GPU)
            build_faiss_index(vector_searcher, embeddings, job_ids)
            vector_searcher.save(INDEX_PATH, ID_MAP_PATH)
    else:
        logging.info("Index files not found. Building and saving index...")
        build_faiss_index(vector_searcher, embeddings, job_ids)
        vector_searcher.save(INDEX_PATH, ID_MAP_PATH)


    # --- 4. Similarity Search & Output ---
    logging.info("--- Step 4: Similarity Search & Output Generation ---")
    if vector_searcher.ntotal == 0:
        logging.warning("Index is empty. Cannot perform search.")
        return

    # Handle Sampling
    num_total_embeddings = len(embeddings)
    query_embeddings_to_search = embeddings
    query_indices_to_map = list(range(num_total_embeddings)) # Ensure it's a list for consistent indexing
    num_queries = num_total_embeddings

    if SEARCH_SAMPLE_SIZE is not None and SEARCH_SAMPLE_SIZE > 0 and SEARCH_SAMPLE_SIZE < num_total_embeddings:
        logging.info(f"Sampling {SEARCH_SAMPLE_SIZE} embeddings for queries...")
        query_indices_to_map = random.sample(range(num_total_embeddings), SEARCH_SAMPLE_SIZE)
        query_embeddings_to_search = embeddings[query_indices_to_map]
        num_queries = len(query_embeddings_to_search)
        logging.info(f"Using {num_queries} sampled embeddings for search.")
    elif SEARCH_SAMPLE_SIZE is not None and SEARCH_SAMPLE_SIZE <= 0:
        logging.warning("SEARCH_SAMPLE_SIZE must be positive. Defaulting to full search.")
        logging.info(f"Performing full search using all {num_queries} embeddings as queries.")
    else: # SEARCH_SAMPLE_SIZE is None or >= num_total_embeddings
        logging.info(f"Performing full search using all {num_queries} embeddings as queries.")


    logging.info(f"Performing nearest neighbor search (k={NEIGHBORS_K}) for {num_queries} items...")
    start_search_time = time.time()
    distances_l2, neighbor_ids_list = vector_searcher.search(query_embeddings_to_search, NEIGHBORS_K)
    end_search_time = time.time()
    logging.info(f"Search completed in {end_search_time - start_search_time:.2f} seconds.")

    logging.info(f"Processing results and filtering by threshold ({SIMILARITY_THRESHOLD})...")
    results = []
    processed_pairs = set()

    for i in range(num_queries):
        original_query_index = query_indices_to_map[i]
        query_id = job_ids[original_query_index]

        if len(neighbor_ids_list[i]) > 1:
            neighbor_id = neighbor_ids_list[i][1]
            neighbor_l2_dist = distances_l2[i][1]
        else:
             logging.warning(f"Query {query_id} (original index {original_query_index}) returned fewer than 2 neighbors.")
             continue

        if neighbor_id is not None:
            if query_id == neighbor_id: continue

            similarity_score = l2_to_cosine_similarity(neighbor_l2_dist)

            if similarity_score >= SIMILARITY_THRESHOLD:
                id1 = min(str(query_id), str(neighbor_id))
                id2 = max(str(query_id), str(neighbor_id))
                pair = (id1, id2)

                if pair not in processed_pairs:
                    results.append({
                        'Job ID 1': id1,
                        'Job ID 2': id2,
                        'Similarity Score': round(similarity_score, 4)
                    })
                    processed_pairs.add(pair)

    logging.info(f"Found {len(results)} pairs above similarity threshold {SIMILARITY_THRESHOLD}.")

    # Create DataFrame and save
    if results:
        results_df = pd.DataFrame(results)
        results_df.sort_values(by='Similarity Score', ascending=False, inplace=True)
        try:
            results_df.to_csv(FINAL_OUTPUT_CSV, index=False)
            logging.info(f"Similarity results saved to '{FINAL_OUTPUT_CSV}'.")
            logging.info("\nSample of results:")
            print(results_df.head(10).to_string())
        except Exception as e:
            logging.error(f"Failed to save results CSV: {e}")
    else:
        logging.info("No pairs found above the similarity threshold.")


    end_pipeline_time = time.time()
    logging.info(f"--- Pipeline Finished ---")
    logging.info(f"Total pipeline execution time: {end_pipeline_time - start_pipeline_time:.2f} seconds")

# --- Run the pipeline ---
if __name__ == "__main__":
    run_similarity_pipeline()