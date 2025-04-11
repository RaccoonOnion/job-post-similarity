# Job Post Similarity Detection

## Video Resource
[[Watch the video]](https://youtu.be/11H3C3HqTAo)
## Objective
This project identifies duplicate or highly similar job postings from a given dataset using text embeddings and vector search techniques. It leverages Natural Language Processing (NLP) and demonstrates software engineering practices through containerization with Docker Compose. The pipeline is designed to be configurable and utilizes caching for intermediate results.

## Table of Contents
1.  [Project Structure](#project-structure)
2.  [Setup and Running](#setup-and-running)
3.  [Configuration](#configuration)
4.  [Pipeline Workflow](#pipeline-workflow)
    * [Data Exploration & Preprocessing](#data-exploration--preprocessing)
    * [Embedding Generation](#embedding-generation)
    * [Vector Search & Indexing](#vector-search--indexing)
    * [Similarity Calculation](#similarity-calculation)
5.  [Evaluation Approach](#evaluation-approach)

## Project Structure
The project is organized as follows:

```text
job-post-similarity/
├── .env                     # Local environment variables (Create from placeholder)
├── Dockerfile               # Defines the Docker image
├── LICENSE                  # Project license file
├── README.md                # Project documentation (This file)
├── analysis_files/          # Output directory for results and analysis plots
│   ├── EDA_preprocess.pdf     # (Optional) PDF export of EDA notebook
│   ├── Embedding Model Choice Justification.pdf # (Optional) Justification document
│   ├── Similarity Threshold Justification.pdf # (Optional) Justification document
│   ├── qual_analysis_first_row.pdf  # (Optional) Example qualitative analysis output
│   ├── qual_analysis_last_row.pdf   # (Optional) Example qualitative analysis output
│   ├── qualitative_analysis_results.csv # Output from evaluation.py (if run)
│   ├── similarity_distribution.png      # Output from evaluation.py (if run)
│   └── similarity_results.csv           # Final output from main.py
├── app/                     # Source code directory
│   ├── EDA_proprocess.ipynb   # Jupyter notebook for EDA
│   ├── evaluation.py          # Script for evaluation methods (qualitative, distribution)
│   ├── fetech_jd.py           # (Potentially for qualitative analysis details)
│   ├── generate_embeddings.py # Module for embedding generation logic
│   ├── main.py                # Main script to run the pipeline
│   ├── preprocess_data.py     # Module for data preprocessing logic
│   └── vector_search.py       # Module containing the VectorSearch class (Faiss wrapper)
├── data/                    # Data directory (mounted as volume)
│   ├── jobs.csv               # Input: Raw job data (Place here before running)
│   ├── jobs_processed.csv     # Output: Processed data
│   ├── job_embeddings.npy     # Output: Generated embeddings
│   ├── job_ids.npy            # Output: IDs corresponding to embeddings
│   ├── faiss_index.index      # Output: Saved Faiss index
│   └── id_map.pkl             # Output: Saved Faiss ID map
├── docker-compose.yml       # Docker Compose configuration file
└── requirements.txt         # Python dependencies
```

## Setup and Running

### Prerequisites
* Docker ([Install Docker](https://docs.docker.com/get-docker/))
* Docker Compose ([Usually included with Docker Desktop](https://docs.docker.com/compose/install/))

### Setup Steps
1.  **Clone Repository:** Get the project code.
    ```bash
    git clone https://github.com/RaccoonOnion/job-post-similarity.git
    cd job-post-similarity
    ```
2.  **Place Data:** Download the `jobs.csv` file and place it inside the `data/` directory. Create the `data/` directory if it doesn't exist.
3.  **Configure Environment:**
    * Edit the `.env` file to set the desired parameters (see [Configuration](#configuration) section below).

### Running the Pipeline
1.  **Build and Run:** Open a terminal in the project's root directory (where `docker-compose.yml` is located) and run:
    ```bash
    docker compose up --build
    ```
    * `--build`: Rebuilds the Docker image if the `Dockerfile` or code in `app/` has changed. Omit this if you only changed the `.env` file or `jobs.csv`.
    * This command starts the `similarity_app` service defined in `docker-compose.yml`.
    * The container runs the `app/main.py` script.
2.  **Process:** The script will perform the following, checking for existing files at each stage:
    * Preprocess `data/jobs.csv` -> `data/jobs_processed.csv` (if needed).
    * Generate embeddings -> `data/job_embeddings.npy`, `data/job_ids.npy` (if needed).
    * Build/Train Faiss index -> `data/faiss_index.index`, `data/id_map.pkl` (if needed).
    * Perform similarity search.
    * Save results -> `analysis_files/similarity_results.csv`.
3.  **Output:** The generated files (`jobs_processed.csv`, `.npy` files, `.index`, `.pkl`, `similarity_results.csv`) will appear in your local `data/` and `analysis_files/` directories due to the volume mounts defined in `docker-compose.yml`. They will persist after the container stops.

## Configuration
Key parameters can be configured by editing the `.env` file before running `docker compose up`:

* `INDEX_DESCRIPTION`: The Faiss index type string (Default: `HNSW32`). Options include `IndexFlatL2` (exact but slow/memory-heavy), `IVF100,Flat` (needs training), `HNSW32` (fast, no training).
* `SIMILARITY_THRESHOLD`: The cosine similarity score threshold to consider jobs duplicates (Default: `0.90`).
* `SEARCH_SAMPLE_SIZE`: Number of jobs to use as queries for the final similarity search. Leave empty or set high (e.g., >91000) to search using all jobs. Set to a lower number (e.g., `10000`) for faster testing/sampling. (Default: `""` - full search).
* `USE_GPU`: Set to `True` to attempt using GPU acceleration (requires `faiss-gpu` in `requirements.txt` and appropriate host setup). Set to `False` to force CPU. (Default: `False`).

## Pipeline Workflow

The `app/main.py` script orchestrates the pipeline:

### Data Exploration & Preprocessing
* **EDA:** Initial analysis was performed in `app/EDA_proprocess.ipynb`. Key findings include the need for HTML parsing, handling minor missing data, removing duplicates based on key fields, and cleaning location data.
* **Preprocessing (`app/preprocess_data.py`):** If `data/jobs_processed.csv` doesn't exist, `main.py` calls this module to:
    1.  Load `data/jobs.csv`.
    2.  Parse HTML from `jobDescRaw` into `jobDescClean`.
    3.  Handle missing values (fill/drop).
    4.  Remove duplicate job postings.
    5.  Clean location fields (`finalState`, `finalZipcode`, `finalCity`).
    6.  Clean text (`jobDescClean`: lowercase, whitespace).
    7.  Drop unused columns.
    8.  Save the result to `data/jobs_processed.csv`.

### Embedding Generation
* **Model (`app/generate_embeddings.py`):** Uses `sentence-transformers` with `all-MiniLM-L6-v2` (configurable via `EMBEDDING_MODEL_NAME` in `main.py`, though not currently an env var). This model is chosen for its balance of speed and performance on semantic similarity tasks.
* **Process (`app/main.py` calls functions from `generate_embeddings.py`):** If `data/job_embeddings.npy` or `data/job_ids.npy` don't exist:
    1.  Loads `data/jobs_processed.csv`.
    2.  Loads the embedding model.
    3.  Encodes the `jobDescClean` text into vectors.
    4.  Saves embeddings to `data/job_embeddings.npy` and corresponding `lid`s to `data/job_ids.npy`.

### Vector Search & Indexing
* **Library (`app/vector_search.py`):** Uses `Faiss` via the `VectorSearch` wrapper class. This class handles index creation (CPU/GPU), training (if needed by index type), adding vectors, searching, saving, and loading.
* **Process (`app/main.py`):**
    1.  Initializes `VectorSearch` with the dimension, `INDEX_DESCRIPTION`, and `USE_GPU` setting from the environment variables.
    2.  Checks if `data/faiss_index.index` and `data/id_map.pkl` exist.
    3.  If yes, loads the existing index and ID map.
    4.  If no, it builds the index:
        * Calls `vector_searcher.train()` if the chosen index type requires it (e.g., `IVF100,Flat`).
        * Calls `vector_searcher.add()` in batches to add all embeddings and IDs.
        * Calls `vector_searcher.save()` to persist the index and ID map to the `data/` directory.

### Similarity Calculation
* **Process (`app/main.py`):**
    1.  Determines the query set (all embeddings or a sample based on `SEARCH_SAMPLE_SIZE`).
    2.  Calls `vector_searcher.search()` to find the top `k=2` neighbors for each query vector in the full index.
    3.  Calculates the cosine similarity between each query and its nearest neighbor (excluding itself).
    4.  Filters the results, keeping only pairs with similarity >= `SIMILARITY_THRESHOLD`.
    5.  Stores unique pairs (Job ID 1, Job ID 2, Similarity Score) ensuring ID1 < ID2.
    6.  Saves the resulting pairs, sorted by similarity, to `analysis_files/similarity_results.csv`.

## Evaluation Approach
* **Ground Truth:** No ground truth labels for duplicates were provided.
* **Methods Used (`app/evaluation.py`):** The evaluation script (run separately or its functions adapted) uses these techniques primarily for analysis and threshold justification:
    1.  **Qualitative Analysis:** Manually inspecting nearest neighbors for a random sample of jobs to gauge relevance (`analysis_files/qualitative_analysis_results.csv`).
    2.  **Similarity Distribution Plot:** Comparing the distribution of similarity scores for nearest neighbors (likely duplicates) vs. random pairs (likely non-duplicates) to inform threshold selection (`analysis_files/similarity_distribution.png`).
* **Threshold:** A cosine similarity threshold of **0.90** (configurable via `.env`) is used in `main.py` to identify potential duplicates based on prior analysis. This prioritizes precision. **Because in real world settings, the cost of false positives (annoying users with too frequent alerts) is higher than false negatives (not able to catch all duplicates).**