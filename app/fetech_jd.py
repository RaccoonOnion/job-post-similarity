import pandas as pd
import numpy as np
import os

# --- Configuration ---
ANALYSIS_DIR = 'analysis_files' # Directory containing analysis files
DATA_DIR = 'data' # Directory containing data files
QUALITATIVE_RESULTS_CSV = f'{ANALYSIS_DIR}/qualitative_analysis_results.csv'
PROCESSED_JOBS_CSV = f'{DATA_DIR}/jobs_processed.csv' # Path to the processed jobs CSV
ID_COLUMN_PROCESSED = 'lid' # Column name for job ID in processed jobs.csv
DESC_COLUMN_PROCESSED = 'jobDescClean' # Column name for cleaned description
# --- MODIFICATION: Define output filenames ---
OUTPUT_FILENAME_FIRST = 'qual_analysis_first_row.md'
OUTPUT_FILENAME_LAST = 'qual_analysis_last_row.md'

# --- Function to safely get IDs from a row ---
def get_ids_from_row(row):
    """Extracts valid Query and Neighbor IDs from a qualitative results row."""
    ids_to_fetch = []
    # Add Query ID
    if pd.notna(row.get('Query_ID')):
        ids_to_fetch.append(str(row['Query_ID'])) # Ensure string type

    # Add Neighbor IDs (assuming columns are Neighbor_1_ID, Neighbor_2_ID, etc.)
    for i in range(1, 6): # Check up to Neighbor_5_ID
        col_name = f'Neighbor_{i}_ID'
        if col_name in row and pd.notna(row[col_name]):
            ids_to_fetch.append(str(row[col_name])) # Ensure string type
    return ids_to_fetch

# --- Function to retrieve descriptions ---
def get_descriptions(ids, df_processed_data):
    """Retrieves descriptions for a list of IDs from the processed dataframe."""
    descriptions = {}
    missing_ids = []
    # Ensure the index is the ID column for quick lookup
    if df_processed_data.index.name != ID_COLUMN_PROCESSED:
         print(f"Warning: Processed DataFrame index is not '{ID_COLUMN_PROCESSED}'. Setting index.")
         try:
            # Use inplace=False to avoid modifying the original df if passed around
            df_processed_data = df_processed_data.set_index(ID_COLUMN_PROCESSED)
         except KeyError:
             print(f"Error: Column '{ID_COLUMN_PROCESSED}' not found in processed CSV.")
             return descriptions, missing_ids


    for job_id in ids:
        try:
            # Use .loc for index lookup
            desc = df_processed_data.loc[job_id, DESC_COLUMN_PROCESSED]
            # Ensure description is a string, handle potential non-string types
            descriptions[job_id] = str(desc) if pd.notna(desc) else "[Description Missing in Processed File]"
        except KeyError:
            print(f"Warning: ID '{job_id}' not found in processed jobs CSV.")
            missing_ids.append(job_id)
            descriptions[job_id] = "[Description Not Found in Processed File]"
        except Exception as e:
            print(f"Error retrieving description for ID '{job_id}': {e}")
            missing_ids.append(job_id)
            descriptions[job_id] = f"[Error Retrieving Description: {e}]"
    return descriptions, missing_ids

# --- Main Script ---
print(f"Loading qualitative results from: {QUALITATIVE_RESULTS_CSV}")
try:
    df_qual = pd.read_csv(QUALITATIVE_RESULTS_CSV)
except FileNotFoundError:
    print(f"Error: File not found '{QUALITATIVE_RESULTS_CSV}'")
    exit()
except Exception as e:
    print(f"Error reading qualitative results CSV: {e}")
    exit()

if df_qual.empty:
    print("Error: Qualitative results CSV is empty.")
    exit()

# Get first and last rows
first_row_data = df_qual.iloc[0].to_dict()
last_row_data = df_qual.iloc[-1].to_dict()

# Extract IDs to fetch
first_row_ids = get_ids_from_row(first_row_data)
last_row_ids = get_ids_from_row(last_row_data)
all_ids_to_fetch = list(set(first_row_ids + last_row_ids)) # Unique IDs

print(f"\nIDs to fetch from first row sample: {first_row_ids}")
print(f"IDs to fetch from last row sample: {last_row_ids}")

# Load processed job data
print(f"\nLoading processed job data from: {PROCESSED_JOBS_CSV}")
try:
    # Load only the ID and Clean Description columns to save memory
    df_processed_data = pd.read_csv(
        PROCESSED_JOBS_CSV,
        usecols=[ID_COLUMN_PROCESSED, DESC_COLUMN_PROCESSED],
        index_col=ID_COLUMN_PROCESSED # Set index during load
    )
    print(f"Loaded processed data for {len(df_processed_data)} jobs.")
except FileNotFoundError:
    print(f"Error: File not found '{PROCESSED_JOBS_CSV}'")
    exit()
except ValueError as e:
     print(f"Error loading processed CSV. Check if columns '{ID_COLUMN_PROCESSED}' and '{DESC_COLUMN_PROCESSED}' exist: {e}")
     # Attempt loading without index_col if setting it failed
     try:
         df_processed_data = pd.read_csv(
             PROCESSED_JOBS_CSV,
             usecols=[ID_COLUMN_PROCESSED, DESC_COLUMN_PROCESSED]
         )
         # Note: get_descriptions function will attempt to set index later if needed
     except Exception as inner_e:
          print(f"Failed to load processed CSV even without index_col: {inner_e}")
          exit()
except Exception as e:
    print(f"Error reading processed jobs CSV: {e}")
    exit()


# Retrieve all required descriptions at once
print(f"\nRetrieving descriptions for {len(all_ids_to_fetch)} unique IDs...")
# Pass the loaded processed dataframe to the function
all_descriptions, _ = get_descriptions(all_ids_to_fetch, df_processed_data)
print("Description retrieval complete.")

# --- Prepare Output Documents ---

# Update formatting function for cleaned text
def format_output(row_data, descriptions):
    output_md = []
    query_id = str(row_data.get('Query_ID', 'N/A'))
    query_title = row_data.get('Query_Title', 'N/A')
    output_md.append(f"# Query Job")
    output_md.append(f"**ID:** {query_id}")
    output_md.append(f"**Title:** {query_title}")
    # Display cleaned description as plain text or within text block
    output_md.append(f"**Description (`jobDescClean`):**\n\n{descriptions.get(query_id, '[Not Found]')}\n")
    output_md.append("\n---\n")

    for i in range(1, 6):
        neighbor_id_col = f'Neighbor_{i}_ID'
        neighbor_title_col = f'Neighbor_{i}_Title'
        neighbor_sim_col = f'Neighbor_{i}_Similarity'

        neighbor_id = row_data.get(neighbor_id_col)
        if pd.isna(neighbor_id):
            continue # Skip if neighbor ID is missing

        neighbor_id = str(neighbor_id)
        neighbor_title = row_data.get(neighbor_title_col, 'N/A')
        neighbor_sim = row_data.get(neighbor_sim_col, 'N/A')

        output_md.append(f"## Neighbor {i}")
        output_md.append(f"**ID:** {neighbor_id}")
        output_md.append(f"**Title:** {neighbor_title}")
        output_md.append(f"**Similarity:** {neighbor_sim:.4f}" if isinstance(neighbor_sim, (float, np.number)) else f"**Similarity:** {neighbor_sim}")
        # Display cleaned description as plain text or within text block
        output_md.append(f"**Description (`jobDescClean`):**\n\n{descriptions.get(neighbor_id, '[Not Found]')}\n")
        output_md.append("\n---\n")

    return "\n".join(output_md)

# Generate markdown for first and last rows
first_row_markdown = format_output(first_row_data, all_descriptions)
last_row_markdown = format_output(last_row_data, all_descriptions)

print("\nMarkdown content generated for first and last row samples.")

# --- MODIFICATION: Write markdown content to files ---
print(f"\nWriting first row sample to: {OUTPUT_FILENAME_FIRST}")
try:
    with open(OUTPUT_FILENAME_FIRST, 'w', encoding='utf-8') as f:
        f.write(first_row_markdown)
    print("Successfully wrote first row sample file.")
except Exception as e:
    print(f"Error writing file {OUTPUT_FILENAME_FIRST}: {e}")

print(f"\nWriting last row sample to: {OUTPUT_FILENAME_LAST}")
try:
    with open(OUTPUT_FILENAME_LAST, 'w', encoding='utf-8') as f:
        f.write(last_row_markdown)
    print("Successfully wrote last row sample file.")
except Exception as e:
    print(f"Error writing file {OUTPUT_FILENAME_LAST}: {e}")

print("\n--- Script Finished ---")