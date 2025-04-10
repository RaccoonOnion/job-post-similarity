# job-post-analysis

conda create -n canaria-mle python=3.10 -y
conda activate canaria-mle
pip install -r requirements.txt

huggingface-cli login

Shape of embeddings array: (95260, 384)