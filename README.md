# What Lives: Definition Clustering Analysis

A hierarchical clustering analysis of expert definitions of life using LLM-generated correlation matrices. This project aims to quantitatively analyze conceptual similarities between definitions and identify emergent semantic clusters.

## Project Overview

What Lives explores the diversity of expert definitions of life by:

1. Collecting definitions from scientists, philosophers, and researchers
2. Using LLMs to generate pairwise correlation scores between definitions
3. Creating correlation matrices based on semantic similarity
4. Applying hierarchical clustering to identify groups of conceptually related definitions
5. Analyzing emergent clusters to identify common themes

## Key Features

- **LLM-Powered Semantic Analysis**: Uses advanced language models to quantify definitional similarities
- **Discipline Vector Generation**: Creates embeddings of definitions within a multidimensional space of academic disciplines
- **Hierarchical Clustering**: Applies agglomerative clustering with dendrogram visualization
- **Optimal Cluster Analysis**: Automatically determines optimal clustering using silhouette scores
- **UMAP Dimensionality Reduction**: For visualizing definitions in a 2D embedding space

## Getting Started

### Prerequisites

- Docker Desktop
- Git
- OpenAI and/or Anthropic API keys

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AttuneIntelligence/what-lives.git
   cd what-lives
   ```

2. Create a `.env` file in the project root with your API keys:

   ```
   ANTHROPIC_API_KEY='your_anthropic_key_here'
   OPENAI_API_KEY='your_openai_key_here'
   ```

3. Launch the Docker container:

   ```bash
   ./run.sh
   ```

4. Start Jupyter Lab from within the container:

   ```bash
   ./bin/jupyter.sh
   ```

5. Access Jupyter Lab at [http://127.0.0.1:8888/](http://127.0.0.1:8888/)

6. Open and run the analysis notebook at `final/final-dev.ipynb`

## Methodology

![Methodology Diagram](https://github.com/AttuneIntelligence/what-lives/raw/main/assets/methodology_diagram.png)

The analysis follows these steps:

1. **Data Collection**: Gathering definitions of life from experts across diverse fields
2. **Correlation Matrix Creation**: Using LLMs to generate pairwise semantic similarity scores
3. **Hierarchical Clustering**: Applying agglomerative clustering to identify related definition groups
4. **Semantic Analysis**: Analyzing identified clusters to extract common themes and conceptual frameworks

## Sample Results

The project identifies distinct conceptual clusters in definitions of life, including:

- Information & Computation-based definitions
- Thermodynamics & Energy-focused perspectives
- Dynamic Systems & Self-Organization approaches
- Structure & Function-centered definitions

## Project Structure

```
what-lives/
├── bin/                      # Shell scripts for environment setup
├── data/                     # Input datasets
├── assets/                   # Figures and diagrams
├── final/                    # Final analysis notebooks
│   └── final-dev.ipynb       # Primary analysis notebook
├── src/                      # Source code
│   ├── what_lives.py         # Main analysis class
│   ├── infer.py              # LLM inference utilities
│   └── prompts/              # LLM system prompts
├── out/                      # Generated outputs
├── .env                      # Environment variables (API keys)
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
└── run.sh                    # Docker launch script
```

## Dependencies

- pandas, numpy, matplotlib
- scipy, scikit-learn, seaborn
- umap-learn
- asyncio, aiohttp
- openai, anthropic

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all the experts who contributed their definitions of life
- Built with assistance from OpenAI and Anthropic language models
