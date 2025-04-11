import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Assuming VectorSearch class is in vector_search.py
try:
    # Use the version with GPU support (vector_search_gpu immersive)
    from vector_search import VectorSearch
except ImportError:
    logging.error("Error: Ensure VectorSearch class is defined in vector_search.py.")
    # If running standalone without the file, you might need to paste the class definition here
    exit()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function ---
def l2_to_cosine_similarity(l2_distance):
    """Converts L2 distance to cosine similarity (assumes normalized embeddings)."""
    similarity = 1 - (np.square(l2_distance) / 2)
    return np.clip(similarity, -1.0, 1.0)

def load_evaluation_data(embeddings_path, ids_path, processed_data_path, id_column_processed):
    """Loads embeddings, IDs, and optional processed data."""
    logging.info("--- Loading Data ---")
    try:
        embeddings = np.load(embeddings_path)
        if embeddings.dtype != np.float32:
            logging.info("Converting embeddings to float32...")
            embeddings = embeddings.astype(np.float32)

        job_ids = np.load(ids_path, allow_pickle=True).tolist()
        logging.info(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")
        logging.info(f"Loaded {len(job_ids)} job IDs.")

        if embeddings.shape[0] != len(job_ids):
            raise ValueError("Mismatch between number of embeddings and number of IDs.")

        df_processed = None
        if processed_data_path and os.path.exists(processed_data_path):
            try:
                df_processed = pd.read_csv(processed_data_path)
                # Ensure the specified ID column exists before setting index
                if id_column_processed in df_processed.columns:
                     df_processed = df_processed.set_index(id_column_processed)
                     logging.info(f"Loaded processed data from {processed_data_path} for context.")
                else:
                    logging.warning(f"ID column '{id_column_processed}' not found in {processed_data_path}. Context lookup might fail.")
                    df_processed = None # Reset to None if index can't be set correctly
            except Exception as e:
                logging.warning(f"Could not load or set index for processed data: {e}")
                df_processed = None
        else:
            logging.warning(f"Processed data file '{processed_data_path}' not found or not specified. Qualitative analysis will lack job title context.")

        return embeddings, job_ids, df_processed

    except FileNotFoundError as e:
        logging.error(f"Error: Required file not found. {e}")
        raise
    except ValueError as e:
         logging.error(f"Data loading error: {e}")
         raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        raise

def build_faiss_index(vector_searcher, embeddings, ids, batch_size=10000):
    """Builds the Faiss index by adding embeddings in batches."""
    logging.info("--- Building Faiss Index ---")
    # Training is handled within VectorSearch based on index type and GPU status
    if not vector_searcher.index.is_trained:
        logging.info("Index requires training. Calling train method...")
        try:
            vector_searcher.train(embeddings) # Pass all embeddings
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

def perform_qualitative_analysis(vector_searcher, embeddings, ids, df_processed, sample_size, k, output_csv):
    """Performs qualitative analysis by checking nearest neighbors for sample jobs."""
    logging.info("--- Performing Qualitative Analysis ---")
    if sample_size <= 0:
        logging.info("Sample size is 0 or less. Skipping qualitative analysis.")
        return []

    if sample_size > vector_searcher.ntotal:
        logging.warning(f"Sample size ({sample_size}) > index size ({vector_searcher.ntotal}). Using index size.")
        sample_size = vector_searcher.ntotal
    if vector_searcher.ntotal == 0:
         logging.warning("Index is empty. Skipping qualitative analysis.")
         return []

    random_indices = random.sample(range(vector_searcher.ntotal), sample_size)
    query_embeddings_sample = embeddings[random_indices]

    logging.info(f"Searching for {k + 1} neighbors for {sample_size} sample jobs...")
    # Use the search method from the VectorSearch class
    distances_l2, neighbor_ids_list = vector_searcher.search(query_embeddings_sample, k + 1)

    qualitative_results = []
    nearest_neighbor_similarities = []
    logging.info("Processing qualitative results...")
    for i, query_idx_in_embeddings in enumerate(random_indices):
        query_id = ids[query_idx_in_embeddings]
        query_title = "N/A"
        if df_processed is not None and query_id in df_processed.index:
            query_title = df_processed.loc[query_id].get('jobTitle', "N/A")

        result_row = {'Query_ID': query_id, 'Query_Title': query_title}
        # neighbor_ids_list already contains original IDs
        query_neighbor_ids = neighbor_ids_list[i]
        query_neighbor_distances = distances_l2[i]

        valid_neighbor_count = 0
        # Iterate from index 1 (skip self) up to k neighbors found
        for j in range(1, min(k + 1, len(query_neighbor_ids))):
            neighbor_id = query_neighbor_ids[j]
            neighbor_distance = query_neighbor_distances[j]

            if neighbor_id is not None: # Check if a valid neighbor was returned
                neighbor_similarity = l2_to_cosine_similarity(neighbor_distance)

                neighbor_title = "N/A"
                if df_processed is not None and neighbor_id in df_processed.index:
                     neighbor_title = df_processed.loc[neighbor_id].get('jobTitle', "N/A")

                result_row[f'Neighbor_{j}_ID'] = neighbor_id
                result_row[f'Neighbor_{j}_Title'] = neighbor_title
                result_row[f'Neighbor_{j}_Similarity'] = round(neighbor_similarity, 4)

                if j == 1: # Store similarity of the actual nearest neighbor
                    nearest_neighbor_similarities.append(neighbor_similarity)
                valid_neighbor_count += 1
            else:
                # Handle cases where search returned None (e.g., k > ntotal)
                 result_row[f'Neighbor_{j}_ID'] = None
                 result_row[f'Neighbor_{j}_Title'] = None
                 result_row[f'Neighbor_{j}_Similarity'] = None


        # Fill remaining columns if fewer than K valid neighbors found
        for j in range(valid_neighbor_count + 1, k + 1):
             result_row[f'Neighbor_{j}_ID'] = None
             result_row[f'Neighbor_{j}_Title'] = None
             result_row[f'Neighbor_{j}_Similarity'] = None

        qualitative_results.append(result_row)

    results_df = pd.DataFrame(qualitative_results)
    try:
        results_df.to_csv(output_csv, index=False)
        logging.info(f"Qualitative analysis results saved to {output_csv}")
    except Exception as e:
        logging.error(f"Error saving qualitative results to CSV: {e}")

    return nearest_neighbor_similarities

def calculate_random_pair_similarity(embeddings, sample_size):
    """Calculates cosine similarity for a sample of random pairs."""
    logging.info(f"Calculating similarities for {sample_size} random pairs...")
    random_pair_similarities = []
    num_vectors = embeddings.shape[0]
    if num_vectors < 2:
        logging.warning("Need at least 2 vectors for random pair analysis. Skipping.")
        return []

    try:
        for _ in range(sample_size):
            idx1, idx2 = random.sample(range(num_vectors), 2)
            sim = cosine_similarity(embeddings[idx1].reshape(1, -1), embeddings[idx2].reshape(1, -1))[0, 0]
            random_pair_similarities.append(sim)
        logging.info("Random pair similarity calculation complete.")
    except Exception as e:
        logging.error(f"Error calculating random pair similarities: {e}")
    return random_pair_similarities

def plot_similarity_distribution(nearest_neighbor_sims, random_pair_sims, output_png):
    """Plots and saves the similarity distributions."""
    logging.info("Plotting similarity distributions...")
    if not nearest_neighbor_sims and not random_pair_sims:
        logging.warning("No similarity data to plot.")
        return

    plot_data_list = []
    if nearest_neighbor_sims:
        plot_data_list.append(pd.DataFrame({'Similarity': nearest_neighbor_sims, 'Type': 'Nearest Neighbor (Likely Duplicate)'}))
    if random_pair_sims:
        plot_data_list.append(pd.DataFrame({'Similarity': random_pair_sims, 'Type': 'Random Pair (Non-Duplicate)'}))

    if not plot_data_list:
        logging.warning("No valid data frames created for plotting.")
        return

    plot_data = pd.concat(plot_data_list, ignore_index=True)

    plt.figure(figsize=(12, 6))
    try:
        sns.histplot(data=plot_data, x='Similarity', hue='Type', bins=50, kde=True, common_norm=False, stat='density')
        plt.title('Distribution of Cosine Similarity Scores')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_png)
        logging.info(f"Similarity distribution plot saved to {output_png}")
        plt.close()
    except Exception as e:
        logging.error(f"Error saving or generating plot: {e}")

def evaluate_similarity(
    embeddings_path='data/job_embeddings.npy',
    ids_path='data/job_ids.npy',
    processed_data_path='data/jobs_processed.csv',
    id_column_processed='lid',
    index_dimension=384,
    # --- MODIFICATION: Changed default index description ---
    index_description='IVF100,Flat',
    # --- ---
    use_gpu=False, # Default to CPU for evaluation script unless specified
    qualitative_sample_size=50,
    neighbors_k=5,
    random_pairs_sample_size=5000,
    qualitative_output_csv='analysis_files/qualitative_analysis_results.csv',
    plot_output_png='analysis_files/similarity_distribution.png'):
    """Main function to orchestrate the evaluation process."""

    logging.info("--- Starting Evaluation Script ---")
    output_dir = os.path.dirname(qualitative_output_csv)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.dirname(plot_output_png)
    if plot_dir: os.makedirs(plot_dir, exist_ok=True)

    try:
        embeddings, job_ids, df_processed = load_evaluation_data(
            embeddings_path, ids_path, processed_data_path, id_column_processed
        )

        # --- MODIFICATION: Pass updated index_description and use_gpu flag ---
        vector_searcher = VectorSearch(
            dimension=index_dimension,
            index_description=index_description, # Use IVF100,Flat (or passed value)
            use_gpu=use_gpu # Pass GPU flag
        )
        # --- ---

        # Build index (will train if using IVF)
        build_faiss_index(vector_searcher, embeddings, job_ids)

        nearest_neighbor_similarities = perform_qualitative_analysis(
            vector_searcher, embeddings, job_ids, df_processed,
            qualitative_sample_size, neighbors_k, qualitative_output_csv
        )

        random_pair_similarities = calculate_random_pair_similarity(
            embeddings, random_pairs_sample_size
        )

        plot_similarity_distribution(
            nearest_neighbor_similarities, random_pair_similarities, plot_output_png
        )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"Evaluation failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during evaluation: {e}")

    logging.info("--- Evaluation Script Finished ---")


# Example execution block
if __name__ == "__main__":
    DATA_DIR = 'data'
    ANALYSIS_DIR = 'analysis_files'

    evaluate_similarity(
        embeddings_path=os.path.join(DATA_DIR, 'job_embeddings.npy'),
        ids_path=os.path.join(DATA_DIR, 'job_ids.npy'),
        processed_data_path=os.path.join(DATA_DIR, 'jobs_processed.csv'),
        qualitative_output_csv=os.path.join(ANALYSIS_DIR, 'qualitative_analysis_results.csv'),
        plot_output_png=os.path.join(ANALYSIS_DIR, 'similarity_distribution.png'),
        # Explicitly set index type here if different from default
        index_description='IVF100,Flat',
        use_gpu=False # Keep False if running on Macbook
        # Keep other parameters as defaults or adjust as needed
    )