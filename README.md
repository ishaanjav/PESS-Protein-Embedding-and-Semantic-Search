# PESS: Protein Embedding & Semantic Search

This repository contains the code and data for my final project for Princeton's *COS 597A: Long Term Memory in AI - Vector Search and Databases*, taught by Edo Liberty and Matthijs Douze.
 
I created a **semantic search program that allows users (e.g. researchers/doctors) to search for proteins using both natural-language descriptions of a protein’s desired function as well as amino acid sequences of the protein.** Example queries include “Protein that interacts with immunoglobulin. Involved with the digestive system.” or a known fragment of the amino acid sequence for a newly-discovered protein still being studied. This can potentially enhance research capabilities by enabling a more intuitive and comprehensive search for proteins. 

I focused on *mus musculus* (the house mouse), creating 2 vector search indices in Pinecone, one for text embeddings (via OpenAI’s Ada model) and the other for amino acid sequence embeddings (via Meta’s ESM model). Both vector search indices with the cosine metric perform well for returning the proteins most relevant to a given search query. I then tested different methods to transform a vector from the text embedding space to the ESM embedding space, as the function of some proteins is still unknown and they lack a description. 

Here is the [link to the presentation](https://github.com/ishaanjav/PESS-Protein-Embedding-and-Semantic-Search/blob/main/Ishaan%20Javali%20-%20PESS%20-%20COS%20597A%20Final%20Presentation.pdf)

## Project Structure
Below are descriptions of the core files in this project:
- **`Testing.ipynb`**: Implements text index, ESM index, and text-to-ESM index search functions and provides all necessary code to issue queries and display results
- **`data.csv`**: Contains Accession, Name, Sequence, Description, and more for 16,985 *mus musculus* proteins.

### Vector Embedding folder
Contains the code for generating text embeddings and ESM embeddings given `data.csv`. Also contains code to upsert the vectors to search indices in Pinecone.
- **`text_embedder.py`**: Python program that uses OpenAI's API for the Ada text embedding model to embed the 14,871 protein descriptions in `data.csv` and creates a new CSV called `text_embeddings.csv` with a column called `ada_embedding` containing 1536-dimensional vectors.
- **`esm_embedder.py`**: Python program that uses Meta's open-sourced ESM model to embed amino acid sequences for all the proteins in `data.csv`. It creates a new CSV called `esm_embeddings.csv` with a column called "esm_650M_embedding", containing 1280-dimensional vectors.
- **`pinecone_upserter.py`**: Python program to upsert vectors (either ESM or text) to Pinecone search index. It includes as metadata, the protein's name and text description.


### Text to ESM Transforms folder
This folder contains the code for learning a transformation matrix that can be matrix multiplied with a text embedding vector to approximate an ESM embedding vector. `V.npy` and `V_pseudoinv.py` contain the learned matrices and can be loaded using `numpy`.

It also contains the code for some of the neural networks that were tested for this task, each with different loss functions. `evaluate_matrix.py` and `evaluate_network.py` can be used for evaluating the approximations against the original ESM vectors over the entire dataset.  

### Scraping folder
- `scraper.py` contains the code used to scrape InterPro's database. First, InterPro's web interface was used to filter for all human-reviewed *mus musculus* proteins and a TSV was downloaded containing the Accession IDs of the proteins. The scraper program uses this TSV --> CSV to then generate `data.csv` which contains the important information (description & sequence) for the proteins.

