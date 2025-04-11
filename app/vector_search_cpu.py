import faiss
import numpy as np
import os
import pickle  # Using pickle for id_map for simplicity, np.save is also an option


class VectorSearch:
    """
    A class to handle vector search operations using Faiss.

    Attributes:
        dimension (int): The dimensionality of the vectors.
        index (faiss.Index): The Faiss index object.
        id_map (list): A list mapping sequential Faiss indices (0, 1, ...)
                       back to original string IDs (e.g., 'lid').
    """

    def __init__(self, dimension, index_description='IndexFlatL2'):
        """
        Initializes the VectorSearch object.

        Args:
            dimension (int): The dimension of the vectors to be indexed (e.g., 384).
            index_description (str, optional): A Faiss index factory string or specific
                                               index name like 'IndexFlatL2'.
                                               Defaults to 'IndexFlatL2' (exact search).
                                               Examples: 'IVF100,Flat', 'HNSW32'.
        Raises:
            ValueError: If dimension is not a positive integer.
            RuntimeError: If Faiss index creation fails.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")

        self.dimension = dimension
        self.id_map = []
        self.index = None  # Initialize index to None

        print(
            f"Initializing VectorSearch with dimension={dimension} and index='{index_description}'")
        try:
            # --- MODIFICATION START ---
            # Handle IndexFlatL2 directly for robustness, use factory for others
            if index_description == 'IndexFlatL2':
                print("Creating IndexFlatL2 directly.")
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                # Use index_factory for other potentially complex descriptions
                print(
                    f"Using index_factory for description: '{index_description}'")
                self.index = faiss.index_factory(
                    self.dimension, index_description)
            # --- MODIFICATION END ---

            print(f"Faiss index created successfully.")
            # Check if the created index has the correct dimension
            if self.index.d != self.dimension:
                # This case should ideally not happen if faiss works correctly
                raise RuntimeError(
                    f"Faiss index created with incorrect dimension: {self.index.d} != {self.dimension}")
        except Exception as e:
            # Add more context to the error
            raise RuntimeError(
                f"Failed to create Faiss index with description '{index_description}'. Original error: {e}")

    def train(self, embeddings):
        """
        Trains the Faiss index, if required by the index type.

        Args:
            embeddings (np.ndarray): A 2D NumPy array of training vectors (n_samples, dimension).

        Raises:
            TypeError: If embeddings is not a NumPy array.
            ValueError: If embeddings array is empty or has incorrect dimensions.
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a NumPy array.")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings must be a 2D array with shape (*, {self.dimension}).")
        if embeddings.shape[0] == 0:
            raise ValueError("Embeddings array cannot be empty for training.")

        # Check if the index type requires training
        # This check is a bit heuristic; specific IVF/PQ types need training.
        # IndexFlatL2, IndexHNSW do not require separate training.
        if not self.index.is_trained:
            # Check if training is actually supported/needed by the index type
            # For many indexes created by factory string (like IVF), training is needed.
            # IndexFlatL2 will report is_trained = True immediately.
            # We attempt training only if Faiss reports it's not trained.
            print(
                f"Index type requires training. Training with {embeddings.shape[0]} vectors...")
            try:
                self.index.train(embeddings)
                print("Index trained successfully.")
            except Exception as e:
                raise RuntimeError(f"Faiss index training failed: {e}")
        else:
            print("Index does not require training or is already trained.")

    def add(self, embeddings, ids):
        """
        Adds a batch of embeddings and their corresponding original IDs to the index.

        Args:
            embeddings (np.ndarray): A 2D NumPy array of embeddings (n_samples, dimension).
            ids (list or np.ndarray): A list or array of original string IDs corresponding
                                      to the embeddings, in the same order.

        Raises:
            TypeError: If embeddings or ids have incorrect types.
            ValueError: If embeddings and ids have mismatched lengths or invalid shapes.
            RuntimeError: If adding vectors to the Faiss index fails.
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a NumPy array.")
        if not isinstance(ids, (list, np.ndarray)):
            raise TypeError("IDs must be a list or NumPy array.")
        # Ensure embeddings are float32, required by Faiss
        if embeddings.dtype != np.float32:
            print("Warning: Embeddings dtype is not float32. Converting...")
            embeddings = embeddings.astype(np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings must be a 2D array with shape (*, {self.dimension}).")
        if len(embeddings) != len(ids):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match number of IDs ({len(ids)}).")
        if len(embeddings) == 0:
            print("Warning: Called add with empty embeddings/ids list. No vectors added.")
            return

        # Ensure index is trained if it needs to be (relevant for IVF etc.)
        if not self.index.is_trained:
            # This check might be relevant if train() wasn't called explicitly when needed.
            print("Warning: Index might require training before adding vectors.")
            # Faiss will likely raise an error internally if adding to an untrained index that needs it.

        print(f"Adding {len(embeddings)} vectors to the index...")
        try:
            self.index.add(embeddings)
            # Extend the id_map only after successfully adding to the index
            self.id_map.extend(list(ids))  # Ensure it's a list
            print(
                f"Vectors added successfully. Index now contains {self.index.ntotal} vectors.")
            print(f"ID map now contains {len(self.id_map)} entries.")
            # Sanity check
            if self.index.ntotal != len(self.id_map):
                print(
                    f"Warning: Mismatch between index total ({self.index.ntotal}) and id_map length ({len(self.id_map)}). Check logic.")

        except Exception as e:
            raise RuntimeError(f"Failed to add vectors to Faiss index: {e}")

    def remove(self, ids_to_remove):
        """
        Removes specified vectors from the index (Not recommended for most Faiss indexes).

        Args:
            ids_to_remove (list): A list of original job IDs (`lid`s) to remove.

        Raises:
            NotImplementedError: This method is not efficiently supported by default
                                 Faiss indexes like IndexFlatL2 or IndexIVFFlat.
        """
        # Direct removal is often inefficient or unsupported in standard Faiss indexes.
        # Rebuilding the index without the removed IDs is the recommended approach.
        # For indexes supporting removal (like using IndexIDMap), this method would need
        # to find the internal Faiss indices corresponding to ids_to_remove and
        # call self.index.remove_ids().
        raise NotImplementedError("Efficient vector removal is not supported by default. "
                                  "Consider rebuilding the index without the vectors to be removed, "
                                  "or use an index type like IndexIDMap which supports removal but adds complexity.")

    def search(self, query_embeddings, k):
        """
        Finds the k nearest neighbors for one or more query embeddings.

        Args:
            query_embeddings (np.ndarray): A 2D NumPy array (n_queries, dimension).
            k (int): The number of nearest neighbors to retrieve for each query.

        Returns:
            tuple: A tuple containing:
                - distances (np.ndarray): The distances to the neighbors (n_queries, k).
                - neighbor_ids (list[list[str]]): A list of lists, where each inner list
                                                  contains the original string IDs (`lid`s)
                                                  of the k nearest neighbors for a query.

        Raises:
            TypeError: If query_embeddings is not a NumPy array or k is not an integer.
            ValueError: If query_embeddings has incorrect shape or k is not positive.
            RuntimeError: If the Faiss search operation fails.
        """
        if not isinstance(query_embeddings, np.ndarray):
            raise TypeError("Query embeddings must be a NumPy array.")
        # Ensure query embeddings are float32
        if query_embeddings.dtype != np.float32:
            print("Warning: Query embeddings dtype is not float32. Converting...")
            query_embeddings = query_embeddings.astype(np.float32)
        if query_embeddings.ndim == 1:  # Handle single query vector
            query_embeddings = query_embeddings.reshape(1, -1)
        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Query embeddings must be a 2D array with shape (*, {self.dimension}).")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if self.index.ntotal == 0:
            print("Warning: Searching an empty index.")
            # Return empty results matching expected structure
            num_queries = query_embeddings.shape[0]
            empty_distances = np.full(
                (num_queries, k), np.inf, dtype=np.float32)
            empty_ids = [[None for _ in range(k)] for _ in range(
                num_queries)]  # Use None for clarity
            return empty_distances, empty_ids

        # Ensure k is not greater than the number of vectors in the index
        effective_k = min(k, self.index.ntotal)
        if effective_k < k:
            print(
                f"Warning: Requested k={k} neighbors, but index only contains {self.index.ntotal} vectors. Searching for {effective_k} neighbors.")

        if effective_k == 0:  # Should be covered by ntotal check, but belt-and-suspenders
            num_queries = query_embeddings.shape[0]
            empty_distances = np.full(
                (num_queries, k), np.inf, dtype=np.float32)
            empty_ids = [[None for _ in range(k)] for _ in range(num_queries)]
            return empty_distances, empty_ids

        print(
            f"Searching for {effective_k} nearest neighbors for {query_embeddings.shape[0]} queries...")
        try:
            distances, indices = self.index.search(
                query_embeddings, effective_k)
            print("Search completed.")

            # Translate Faiss indices back to original IDs
            neighbor_ids = []
            for query_indices in indices:
                query_neighbor_ids = []
                for idx in query_indices:
                    # Faiss uses -1 if fewer than k neighbors are found (or on error)
                    if idx == -1:
                        query_neighbor_ids.append(None)
                    elif 0 <= idx < len(self.id_map):
                        query_neighbor_ids.append(self.id_map[idx])
                    else:
                        print(
                            f"Warning: Found invalid index {idx} during search result mapping.")
                        query_neighbor_ids.append(None)  # Invalid index

                # Pad with None if fewer than k results were returned by search
                while len(query_neighbor_ids) < k:
                    query_neighbor_ids.append(None)
                neighbor_ids.append(query_neighbor_ids)

            # Pad distances array if needed
            if distances.shape[1] < k:
                padded_distances = np.full(
                    (distances.shape[0], k), np.inf, dtype=np.float32)
                padded_distances[:, :distances.shape[1]] = distances
                distances = padded_distances

            return distances, neighbor_ids

        except Exception as e:
            raise RuntimeError(f"Faiss search failed: {e}")

    def save(self, index_path, id_map_path):
        """
        Saves the Faiss index and the ID mapping to disk.

        Args:
            index_path (str): File path to save the Faiss index (e.g., 'my_index.faiss').
            id_map_path (str): File path to save the ID map (e.g., 'id_map.pkl').

        Raises:
            RuntimeError: If saving the index or ID map fails.
        """
        if not self.index:
            raise RuntimeError("Cannot save: Index is not initialized.")

        print(f"Saving Faiss index to: {index_path}")
        try:
            faiss.write_index(self.index, index_path)
            print("Index saved successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to save Faiss index to {index_path}: {e}")

        print(f"Saving ID map to: {id_map_path}")
        try:
            # Using pickle to save the list of potentially string IDs
            with open(id_map_path, 'wb') as f:
                pickle.dump(self.id_map, f)
            # # Alternative using numpy (might be safer if only dealing with basic types)
            # np.save(id_map_path, np.array(self.id_map, dtype=object))
            print("ID map saved successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to save ID map to {id_map_path}: {e}")

    def load(self, index_path, id_map_path):
        """
        Loads a Faiss index and ID mapping from disk.

        Args:
            index_path (str): File path to load the Faiss index from.
            id_map_path (str): File path to load the ID map from.

        Returns:
            bool: True if loading was successful.

        Raises:
            FileNotFoundError: If index or ID map file does not exist.
            RuntimeError: If loading fails or loaded index dimension mismatch.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Faiss index file not found: {index_path}")
        if not os.path.exists(id_map_path):
            raise FileNotFoundError(f"ID map file not found: {id_map_path}")

        print(f"Loading Faiss index from: {index_path}")
        try:
            self.index = faiss.read_index(index_path)
            # Update dimension based on loaded index
            self.dimension = self.index.d
            print(
                f"Index loaded successfully. Dimension={self.dimension}, NumVectors={self.index.ntotal}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Faiss index from {index_path}: {e}")

        print(f"Loading ID map from: {id_map_path}")
        try:
            # Using pickle to load
            with open(id_map_path, 'rb') as f:
                self.id_map = pickle.load(f)
            # # Alternative using numpy
            # self.id_map = np.load(id_map_path, allow_pickle=True).tolist()
            print(
                f"ID map loaded successfully. Map contains {len(self.id_map)} entries.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ID map from {id_map_path}: {e}")

        # Sanity check
        if self.index.ntotal != len(self.id_map):
            # Allow loading even if mismatch, but print strong warning
            print(
                f"CRITICAL WARNING: Loaded index total ({self.index.ntotal}) does not match loaded id_map length ({len(self.id_map)}). Search results may be incorrect!")
            # Depending on requirements, you might want to raise an error here:
            # raise ValueError("Mismatch between loaded index size and ID map size.")

        return True

    @property
    def ntotal(self):
        """Returns the total number of vectors currently in the index."""
        return self.index.ntotal if self.index else 0


# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # This block demonstrates basic usage.
    # Replace with actual loading of your embeddings and IDs.

    D = 384  # Dimension of embeddings (e.g., all-MiniLM-L6-v2)
    NUM_VECTORS = 1000
    K_NEIGHBORS = 5

    # --- Create dummy data ---
    print("\n--- Example Usage ---")
    print("Generating dummy data...")
    # Ensure dummy data is float32
    dummy_embeddings = np.float32(np.random.random((NUM_VECTORS, D)))
    dummy_ids = [f'job_{i:04d}' for i in range(NUM_VECTORS)]
    dummy_queries = np.float32(np.random.random((5, D)))  # 5 query vectors

    # --- Initialize VectorSearch ---
    try:
        # Using default 'IndexFlatL2' which doesn't need training
        # Uses IndexFlatL2 by default
        vector_searcher = VectorSearch(dimension=D)

        # --- Add data ---
        # Add in batches (e.g., 2 batches of 500)
        batch_size = 500
        for i in range(0, NUM_VECTORS, batch_size):
            print(f"\nAdding batch {i//batch_size + 1}...")
            emb_batch = dummy_embeddings[i:i+batch_size]
            id_batch = dummy_ids[i:i+batch_size]
            vector_searcher.add(emb_batch, id_batch)

        print(f"\nTotal vectors in index: {vector_searcher.ntotal}")

        # --- Search ---
        print(f"\nSearching for {K_NEIGHBORS} neighbors...")
        distances, neighbor_ids = vector_searcher.search(
            dummy_queries, k=K_NEIGHBORS)

        print("\nSearch Results (Distances):")
        print(distances)
        print("\nSearch Results (Neighbor IDs):")
        for i, neighbors in enumerate(neighbor_ids):
            print(f"Query {i}: {neighbors}")

        # --- Save Index ---
        INDEX_FILE = 'example.faiss'
        ID_MAP_FILE = 'example_id_map.pkl'
        print(f"\nSaving index to {INDEX_FILE} and map to {ID_MAP_FILE}...")
        vector_searcher.save(INDEX_FILE, ID_MAP_FILE)

        # --- Load Index ---
        print("\nLoading index from files...")
        # Initialize with a placeholder dimension, load() will update it
        loaded_searcher = VectorSearch(dimension=1)
        load_success = loaded_searcher.load(INDEX_FILE, ID_MAP_FILE)

        if load_success:
            print(f"Loaded index contains {loaded_searcher.ntotal} vectors.")
            # Verify search works on loaded index
            print("\nSearching again using loaded index...")
            distances_loaded, neighbor_ids_loaded = loaded_searcher.search(
                dummy_queries, k=K_NEIGHBORS)
            # Simple check if results seem consistent (IDs should match)
            print("\nComparing neighbor IDs from original and loaded search:")
            # Need to handle potential None values for comparison
            original_neighbors_str = [
                [str(nid) for nid in sublist] for sublist in neighbor_ids]
            loaded_neighbors_str = [[str(nid) for nid in sublist]
                                    for sublist in neighbor_ids_loaded]
            print(f"Match: {original_neighbors_str == loaded_neighbors_str}")

        # Clean up dummy files
        print("\nCleaning up example files...")
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(ID_MAP_FILE):
            os.remove(ID_MAP_FILE)
        print("Cleanup complete.")

    except Exception as e:
        print(f"\nAn error occurred during example usage: {e}")
        import traceback
        traceback.print_exc()
