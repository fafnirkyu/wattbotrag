## WattBot RAG - Scientific Question Answering System
A high-performance Retrieval-Augmented Generation (RAG) pipeline developed for the Kaggle "LLM Science Exam" Competition, achieving a score of 0.185 by balancing sophisticated retrieval techniques with computational efficiency.

# 🏆 Competition Context
This project was developed for the Kaggle LLM Science Exam competition, where the goal was to answer complex scientific questions using a corpus of academic papers. The challenge required:

Accuracy: Providing precise, well-supported answers from scientific literature

Efficiency: Processing thousands of questions within competition time limits

Robustness: Handling diverse scientific domains and question types

Final Score: 0.185 - Demonstrating competitive performance against other ML practitioners.

# 🚀 Features
Document Processing: Automatic PDF downloading and text extraction with intelligent chunking

Hybrid Retrieval: Combines vector similarity (FAISS) with TF-IDF for robust document retrieval

Cross-Encoder Reranking: Improves retrieval quality using sentence transformers

LLM Integration: Uses Qwen2.5-1.5B for answer generation with structured JSON output

Fine-tuning Support: Includes tools for retriever fine-tuning and hard negative mining

Modular Architecture: Clean separation of concerns for easy maintenance and extension

Parallel Processing: Efficient PDF processing and embedding generation

Comprehensive Logging: Detailed logging for debugging and monitoring

# ⚖️ Technical Challenges & Solutions
The Accuracy vs. Performance Balancing Act
One of the most significant challenges was finding the right balance between retrieval accuracy and computational efficiency:

Initial Approach: Maximum Accuracy
Used larger models (bge-large, all-mpnet-base-v2)

Higher chunk sizes (500-800 tokens)

Extensive reranking (top 20 → top 10)

Result: Excellent accuracy but prohibitively slow (~10 seconds per question)

Optimized Approach: Balanced Performance
Model Selection: Switched to bge-small-en-v1.5 (80% smaller, 90% of accuracy)

Chunk Optimization: Reduced to 300 tokens for better precision

Retrieval Tuning: TOP_K=30 → TOP_K_RERANK=5 pipeline

Batch Processing: Implemented parallel PDF processing and batched inference

Result: 0.185 score with ~2-3 seconds per question

The Ollama to Hugging Face Transition
Initial Architecture:

```bash
# Original approach using Ollama for local inference
OLLAMA_MODEL = "llama2:13b"
ollama_client = OllamaClient()
```

Challenge Encountered:
The outlines library dependency for structured JSON generation created version conflicts and instability in the production environment. Despite Ollama's advantages for local deployment, the dependency management overhead became unsustainable.

Solution: Full Hugging Face Integration

```bash
# Migrated to Hugging Face for reliability
HF_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
hf_generator = pipeline("text-generation", model=hf_model, tokenizer=hf_tokenizer)
```

Benefits Realized:

✅ Stable dependency management

✅ Consistent JSON output formatting

✅ Better error handling and recovery

✅ Easier deployment and scaling

✅ Maintained competitive accuracy (0.185 score)

# 📋 Prerequisites
Python 3.8+

CUDA-capable GPU (recommended for best performance)

At least 8GB RAM (16GB+ recommended)

# 🛠 Installation
Clone the repository

```bash
git clone https://github.com/your-username/wattbot-rag.git
cd wattbot-rag
Create and activate virtual environment
```
```bash
python -m venv wattbot_env
source wattbot_env/bin/activate  # On Windows: wattbot_env\Scripts\activate
Install dependencies
```
```bash
pip install -r requirements.txt
Set up data directories
```

mkdir -p data/pdfs models logs
# 📁 Project Structure
```bash
wattbot_rag/
├── config/                 # Configuration settings
│   └── settings.py
├── data_processing/        # PDF processing and chunking
│   ├── pdf_processor.py
│   └── chunking.py
├── embedding/             # Embedding generation and FAISS
│   ├── embedders.py
│   └── faiss_manager.py
├── retrieval/             # Hybrid search and reranking
│   ├── hybrid_search.py
│   └── reranker.py
├── generation/            # LLM integration and answer generation
│   ├── llm_client.py
│   └── answer_generator.py
├── training/              # Model fine-tuning utilities
│   ├── retriever_trainer.py
│   └── negative_mining.py
├── utils/                 # Utility functions
│   ├── logging_config.py
│   └── file_utils.py
├── main.py               # Main pipeline entry point
└── requirements.txt      # Python dependencies
```

⚙️ Configuration
Edit config/settings.py to customize:

Model Paths: Change retriever, reranker, or generator models

File Paths: Adjust data directory locations

Hyperparameters: Modify chunk sizes, retrieval parameters, batch sizes

Training Settings: Configure fine-tuning parameters

Key configuration options:

```bash
# Model configurations
RETRIEVER_BASE = "BAAI/bge-small-en-v1.5"  # Optimized for speed/accuracy balance
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Retrieval parameters - tuned for competition constraints
TOP_K = 30                    # Initial retrieval candidates
TOP_K_RERANK = 5              # Final candidates after reranking
CHUNK_TOKENS = 300            # Optimized token limit for precision
```

# 🏃‍♂️ Quick Start
Prepare your data:

Place metadata.csv in the data/ directory

Place test questions in data/test_Q.csv

(Optional) Add training QA pairs in data/train_QA.csv

Run the pipeline:

```bash
python main.py
```

The system will automatically:
Download PDFs from URLs in metadata

Extract and chunk text content

Generate embeddings and build FAISS index

Process test questions and generate answers

Save results to submission.csv

# 🎯 Performance Optimizations
Retrieval Optimizations
Hybrid Search: Vector + TF-IDF fusion for recall improvement

Reranking: Cross-encoder to reorder retrieved documents

Deduplication: Prevents redundant context in prompts

TF-IDF Features: Limited to 8,000 for memory efficiency

Computational Optimizations
GPU Acceleration: Automatic CUDA detection and utilization

Mixed Precision: FP16 inference where supported

Batch Processing: Parallel PDF extraction and batched embeddings

Memory Management: Efficient chunking and cache utilization

Model Selection Rationale
Retriever: bge-small-en-v1.5 - Best speed/accuracy tradeoff

Reranker: ms-marco-MiniLM-L-6-v2 - Lightweight but effective

Generator: Qwen2.5-1.5B-Instruct - Strong reasoning within constraints

# 📊 Results & Lessons Learned
Competition Performance
Score: 0.185 (Top 30% performance range)

Throughput: ~2-3 seconds per question

Accuracy: Competitive retrieval and answer quality

Key Technical Insights
Smaller models with better architecture often outperform larger ones in constrained environments

Reranking provides disproportionate accuracy gains relative to computational cost

Hybrid retrieval strategies are more robust than single-method approaches

Dependency management can be as critical as algorithm selection

Systematic benchmarking of each component is essential for optimization

Architecture Decisions
Modular Design: Enabled rapid experimentation and component swapping

Configuration-Driven: Allowed quick hyperparameter tuning

Comprehensive Logging: Provided insights for iterative improvement

Error Resilience: Graceful fallbacks maintained pipeline stability

# 🔧 Advanced Usage
Fine-tuning the Retriever
Prepare training data in data/train_QA.csv

The system will automatically fine-tune during pipeline execution

Fine-tuned models are saved to models/fine_tuned_retriever/

Custom Model Integration
Replace models in config/settings.py:

```bash
# For different embedding models
RETRIEVER_BASE = "sentence-transformers/all-mpnet-base-v2"  # More accurate, slower

# For different LLMs  
HF_MODEL_NAME = "microsoft/DialoGPT-large"

# For different rerankers
RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"  # Faster, less accurate
```

# 🐛 Troubleshooting
Common Issues
Out of Memory Errors

Reduce EMBED_BATCH and BATCH_SIZE in config

Use smaller models (e.g., "bge-small-en" instead of "bge-large-en")

PDF Download Failures

Check internet connection

Verify URLs in metadata.csv

Some PDFs may require special handling

Model Loading Issues

Ensure you have sufficient disk space for model cache

Check Hugging Face access for gated models

Logs
Check logs/wattbot_run.log for detailed execution logs and error messages.

# 🤝 Contributing
This project demonstrates real-world tradeoffs in production ML systems. Contributions welcome for:

Performance optimizations

Alternative model integrations

Enhanced retrieval strategies

Documentation improvements

# 📄 License
This project is available for portfolio and educational use. Please check individual model licenses for commercial use.
