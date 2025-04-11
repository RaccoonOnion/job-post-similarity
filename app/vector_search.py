import faiss
import numpy as np
import os
import pickle
import logging  # Added logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class VectorSearch:
    """
    A class to handle vector search operations using Faiss, with optional GPU support.
    """

    def __init__(self, dimension, index_description='IndexFlatL2', use_gpu=False):
        """
        Initializes the VectorSearch object.

        Args:
            dimension (int): The dimension of the vectors.
            index_description (str, optional): Faiss index factory string. Defaults to 'IndexFlatL2'.
            use_gpu (bool, optional): Whether to attempt using the GPU. Defaults to False.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")

        self.dimension = dimension
        self.id_map = []
        self.index = None
        self.gpu_index = None  # Store GPU index separately if used
        self.use_gpu = use_gpu
        self._gpu_resource = None  # Store GPU resource object

        logging.info(
            f"Initializing VectorSearch with dimension={dimension}, index='{index_description}', use_gpu={use_gpu}")
        try:
            # Create CPU index first
            if index_description == 'IndexFlatL2':
                logging.info("Creating CPU IndexFlatL2 directly.")
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                logging.info(
                    f"Using index_factory for CPU description: '{index_description}'")
                self.index = faiss.index_factory(
                    self.dimension, index_description)

            if self.index.d != self.dimension:
                raise RuntimeError(
                    f"CPU index created with incorrect dimension: {self.index.d} != {self.dimension}")

            # Attempt to move to GPU if requested and possible
            if self.use_gpu:
                try:
                    # Initialize GPU resources (needed for multiple GPUs, good practice)
                    self._gpu_resource = faiss.StandardGpuResources()
                    logging.info("Initialized GPU resources.")
                    # Transfer the CPU index to GPU
                    self.gpu_index = faiss.index_cpu_to_gpu(
                        self._gpu_resource, 0, self.index)  # Use GPU device 0
                    logging.info("Successfully transferred index to GPU.")
                    # Point self.index to the GPU index for operations
                    self.index = self.gpu_index  # Now self.index refers to the GPU version
                except AttributeError:
                    logging.warning(
                        "Faiss GPU extensions not available (faiss.StandardGpuResources or index_cpu_to_gpu). Falling back to CPU.")
                    self.use_gpu = False  # Disable GPU flag if failed
                    self.gpu_index = None
                    self._gpu_resource = None
                except Exception as gpu_e:
                    logging.warning(
                        f"Failed to initialize or transfer index to GPU: {gpu_e}. Falling back to CPU.")
                    self.use_gpu = False  # Disable GPU flag if failed
                    self.gpu_index = None
                    self._gpu_resource = None

            logging.info(
                f"Faiss index initialization complete (using {'GPU' if self.use_gpu else 'CPU'}).")

        except Exception as e:
            raise RuntimeError(
                f"Failed to create Faiss index. Original error: {e}")

    def train(self, embeddings):
        """ Trains the Faiss index (CPU or GPU). """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a NumPy array.")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embeddings shape mismatch.")
        if embeddings.shape[0] == 0:
            raise ValueError("Embeddings array cannot be empty for training.")

        target_index = self.index  # Use the current index (CPU or GPU)

        if not target_index.is_trained:
            logging.info(
                f"Index type requires training. Training with {embeddings.shape[0]} vectors on {'GPU' if self.use_gpu else 'CPU'}...")
            try:
                target_index.train(embeddings)
                logging.info("Index trained successfully.")
            except Exception as e:
                raise RuntimeError(f"Faiss index training failed: {e}")
        else:
            logging.info(
                "Index does not require training or is already trained.")

    def add(self, embeddings, ids):
        """ Adds embeddings and IDs to the index (CPU or GPU). """
        # Type/shape checks... (same as before)
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be NP array.")
        if not isinstance(ids, (list, np.ndarray)):
            raise TypeError("IDs must be list or NP array.")
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError("Embeddings shape")
        if len(embeddings) != len(ids):
            raise ValueError("Embeddings/IDs length mismatch")
        if len(embeddings) == 0:
            return

        target_index = self.index  # Use current index (CPU or GPU)

        if not target_index.is_trained:
            logging.warning(
                "Index might require training before adding vectors.")

        logging.info(
            f"Adding {len(embeddings)} vectors to the {'GPU' if self.use_gpu else 'CPU'} index...")
        try:
            target_index.add(embeddings)
            self.id_map.extend(list(ids))
            logging.info(
                f"Vectors added. Index now contains {target_index.ntotal} vectors.")
            if target_index.ntotal != len(self.id_map):
                logging.warning(
                    f"Index total ({target_index.ntotal}) / id_map ({len(self.id_map)}) mismatch.")
        except Exception as e:
            raise RuntimeError(f"Failed to add vectors to Faiss index: {e}")

    def search(self, query_embeddings, k):
        """ Finds k nearest neighbors (CPU or GPU). """
        # Type/shape checks... (same as before)
        if not isinstance(query_embeddings, np.ndarray):
            raise TypeError("Query must be NP array.")
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != self.dimension:
            raise ValueError("Query shape")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be positive int.")

        target_index = self.index  # Use current index (CPU or GPU)

        if target_index.ntotal == 0:
            logging.warning("Searching an empty index.")
            num_queries = query_embeddings.shape[0]
            empty_distances = np.full(
                (num_queries, k), np.inf, dtype=np.float32)
            empty_ids = [[None] * k for _ in range(num_queries)]
            return empty_distances, empty_ids

        effective_k = min(k, target_index.ntotal)
        if effective_k < k:
            logging.warning(
                f"Requested k={k}, index has {target_index.ntotal}. Searching for {effective_k}.")
        if effective_k == 0:
            # Handle case where index becomes empty after check
            num_queries = query_embeddings.shape[0]
            empty_distances = np.full(
                (num_queries, k), np.inf, dtype=np.float32)
            empty_ids = [[None] * k for _ in range(num_queries)]
            return empty_distances, empty_ids

        logging.info(
            f"Searching for {effective_k} neighbors for {query_embeddings.shape[0]} queries on {'GPU' if self.use_gpu else 'CPU'}...")
        try:
            distances, indices = target_index.search(
                query_embeddings, effective_k)
            logging.info("Search completed.")

            # Translate indices back to original IDs... (same logic as before)
            neighbor_ids = []
            for query_indices in indices:
                query_neighbor_ids = [self.id_map[idx] if idx != -1 and 0 <=
                                      idx < len(self.id_map) else None for idx in query_indices]
                while len(query_neighbor_ids) < k:
                    query_neighbor_ids.append(None)  # Pad if needed
                neighbor_ids.append(query_neighbor_ids)

            # Pad distances array if needed... (same logic as before)
            if distances.shape[1] < k:
                padded_distances = np.full(
                    (distances.shape[0], k), np.inf, dtype=np.float32)
                padded_distances[:, :distances.shape[1]] = distances
                distances = padded_distances

            return distances, neighbor_ids

        except Exception as e:
            raise RuntimeError(f"Faiss search failed: {e}")

    def save(self, index_path, id_map_path):
        """ Saves the index (moves from GPU to CPU if necessary) and ID map. """
        if not self.index:
            raise RuntimeError("Cannot save: Index not initialized.")

        index_to_save = self.index
        if self.use_gpu and self.gpu_index:
            logging.info("Moving index from GPU to CPU for saving...")
            try:
                index_to_save = faiss.index_gpu_to_cpu(self.gpu_index)
                logging.info("Index moved to CPU.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to move index from GPU to CPU for saving: {e}")
        elif self.use_gpu and not self.gpu_index:
            logging.warning(
                "use_gpu is True, but no GPU index found. Saving the CPU index.")
            index_to_save = self.index  # Should already be the CPU index

        logging.info(f"Saving Faiss index to: {index_path}")
        try:
            faiss.write_index(index_to_save, index_path)
            logging.info("Index saved successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to save Faiss index: {e}")

        logging.info(f"Saving ID map to: {id_map_path}")
        try:
            with open(id_map_path, 'wb') as f:
                pickle.dump(self.id_map, f)
            logging.info("ID map saved successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to save ID map: {e}")

    def load(self, index_path, id_map_path):
        """ Loads index and ID map, optionally moving index to GPU. """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(id_map_path):
            raise FileNotFoundError(f"ID map file not found: {id_map_path}")

        logging.info(f"Loading Faiss index from: {index_path}")
        try:
            cpu_index = faiss.read_index(index_path)
            self.dimension = cpu_index.d  # Update dimension from loaded index
            logging.info(
                f"CPU index loaded. Dimension={self.dimension}, NumVectors={cpu_index.ntotal}")

            # Decide where the final index should reside
            if self.use_gpu:
                try:
                    if not self._gpu_resource:  # Initialize resource if not already done
                        self._gpu_resource = faiss.StandardGpuResources()
                    self.gpu_index = faiss.index_cpu_to_gpu(
                        self._gpu_resource, 0, cpu_index)
                    self.index = self.gpu_index  # Point self.index to GPU version
                    logging.info(
                        "Successfully transferred loaded index to GPU.")
                except Exception as gpu_e:
                    logging.warning(
                        f"Failed to move loaded index to GPU: {gpu_e}. Using CPU index.")
                    self.use_gpu = False  # Fallback
                    self.index = cpu_index
                    self.gpu_index = None
            else:
                self.index = cpu_index  # Use the CPU index directly
                self.gpu_index = None
                self._gpu_resource = None

        except Exception as e:
            raise RuntimeError(f"Failed to load Faiss index: {e}")

        logging.info(f"Loading ID map from: {id_map_path}")
        try:
            with open(id_map_path, 'rb') as f:
                self.id_map = pickle.load(f)
            logging.info(
                f"ID map loaded successfully. Map contains {len(self.id_map)} entries.")
        except Exception as e:
            raise RuntimeError(f"Failed to load ID map: {e}")

        # Sanity check
        if self.index.ntotal != len(self.id_map):
            logging.critical(
                f"CRITICAL WARNING: Loaded index total ({self.index.ntotal}) != loaded id_map length ({len(self.id_map)}).")
            # raise ValueError("Mismatch between loaded index size and ID map size.") # Optional: make it fatal

        return True

    @property
    def ntotal(self):
        """Returns the total number of vectors currently in the index."""
        return self.index.ntotal if self.index else 0


# --- Example Usage ---
# (Keep the __main__ block as is for testing the class standalone if needed)
if __name__ == '__main__':
    # Example remains largely the same, but you could test GPU:
    D = 384
    try:
        # Test GPU if available, otherwise fallback
        gpu_available = False
        try:
            res = faiss.StandardGpuResources()  # Test if GPU resources can be initialized
            gpu_available = True
            logging.info("GPU seems available.")
        except AttributeError:
            logging.info("GPU not available (AttributeError).")
        except Exception as e:
            logging.info(f"GPU check failed: {e}")

        searcher_gpu = VectorSearch(dimension=D, use_gpu=gpu_available)
        # ... rest of the example usage from the original file ...
        # You would run the add, search, save, load steps using searcher_gpu
        # The class now handles the GPU logic internally based on the use_gpu flag.

    except Exception as e:
        logging.error(f"\nAn error occurred during example usage: {e}")
        import traceback
        traceback.print_exc()
