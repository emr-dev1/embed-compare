# Embedding Model Comparison Tool

A comprehensive tool for comparing two embedding models using the DeepEval framework, providing detailed metrics and CSV reports.

## Features

- **Comprehensive Metrics**: Combines embedding-specific metrics with DeepEval's evaluation metrics
- **Multiple Input Formats**: Supports CSV, NumPy, and JSON input formats
- **CSV Reports**: Generates detailed CSV reports for easy analysis and sharing
- **Flexible Comparison**: Handles different embedding dimensions and model types
- **Statistical Analysis**: Provides embedding space statistics and similarity analysis

## Installation

```bash
pip install deepeval pandas numpy scikit-learn
```

## Quick Start

### 1. Basic Usage with Generated Data

```python
from embedding_comparison import EmbeddingModelComparator
import numpy as np

# Your data
texts = ["What is AI?", "How does ML work?", "Explain neural networks"]
embeddings1 = np.random.rand(3, 768)   # Model 1: 768-dim embeddings
embeddings2 = np.random.rand(3, 1536)  # Model 2: 1536-dim embeddings

# Run comparison
comparator = EmbeddingModelComparator("BERT-Base", "OpenAI-Ada-002")
results = comparator.run_comprehensive_evaluation(
    texts=texts,
    embeddings1=embeddings1,
    embeddings2=embeddings2,
    create_csv=True,
    output_dir="results"
)
```

### 2. Usage with Real Data Files

```python
from embedding_comparison import (
    EmbeddingModelComparator, 
    load_embeddings_from_csv,
    create_sample_data_files
)
import pandas as pd

# Create sample files (for testing)
create_sample_data_files()

# Load your embeddings
embeddings1 = load_embeddings_from_csv('model1_embeddings.csv')
embeddings2 = load_embeddings_from_csv('model2_embeddings.csv')

# Load text data
df = pd.read_csv('texts.csv')
texts = df['text'].tolist()
contexts = df['context'].tolist()
expected_outputs = df['expected_output'].tolist()

# Run comparison
comparator = EmbeddingModelComparator("My-Model-1", "My-Model-2")
results = comparator.run_comprehensive_evaluation(
    texts=texts,
    embeddings1=embeddings1,
    embeddings2=embeddings2,
    contexts=contexts,
    expected_outputs=expected_outputs,
    create_csv=True
)
```

## Input Data Formats

### Required Data

1. **Embeddings** (numpy arrays):
   ```python
   # Shape: (n_samples, embedding_dimension)
   model1_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
   model2_embeddings = np.array([[0.7, 0.8], [0.9, 1.0]])  # Different dimensions OK
   ```

2. **Texts** (list of strings):
   ```python
   texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
   ```

### Optional Data (Recommended for Better Metrics)

3. **Contexts** (list of strings):
   ```python
   contexts = ["Context for text 1", "Context for text 2", "Context for text 3"]
   ```

4. **Expected Outputs** (list of strings):
   ```python
   expected_outputs = ["Expected answer 1", "Expected answer 2", "Expected answer 3"]
   ```

## Supported File Formats

### CSV Format
```csv
# embeddings_model1.csv
0,1,2,3,...,767,text,context
0.1,0.2,0.3,0.4,...,0.8,"What is AI?","AI context..."
0.2,0.3,0.4,0.5,...,0.9,"How does ML work?","ML context..."
```

### NumPy Format
```python
# Save embeddings
np.save('model1_embeddings.npy', embeddings_array)

# Load embeddings
embeddings = load_embeddings_from_npy('model1_embeddings.npy')
```

### JSON Format
```json
{
  "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
  "texts": ["Sample text 1", "Sample text 2"],
  "contexts": ["Context 1", "Context 2"],
  "model_name": "BERT-Base",
  "embedding_dim": 768
}
```

## Loading Data Functions

```python
# From CSV
embeddings = load_embeddings_from_csv('file.csv', embedding_columns=['0', '1', '2'])

# From NumPy
embeddings = load_embeddings_from_npy('embeddings.npy')

# From JSON
embeddings = load_embeddings_from_json('data.json', embedding_key='embeddings')
```

## Output Files

The tool generates 4 CSV reports in your specified output directory:

### 1. `model_comparison_[timestamp].csv`
High-level comparison between models:
- Model names and basic info
- All metric scores side-by-side
- Embedding statistics
- Easy overview for decision making

### 2. `detailed_metrics_[timestamp].csv`
Detailed metric breakdown:
- Individual metric scores
- Score differences
- Winner for each metric
- Metric categories

### 3. `embedding_statistics_[timestamp].csv`
Embedding space analysis:
- Dimensionality information
- Norm statistics
- Cosine similarity patterns
- Space comparison metrics

### 4. `per_sample_results_[timestamp].csv`
Individual sample analysis:
- Per-text results
- Embedding norms
- Similarity scores

## Available Metrics

### DeepEval Metrics
- **Answer Relevancy**: How relevant responses are to inputs
- **Contextual Relevancy**: Relevance considering context
- **Faithfulness**: Accuracy to source material
- **Bias Detection**: Potential bias in responses
- **Toxicity Detection**: Safety and appropriateness

### Embedding-Specific Metrics
- **Cosine Similarity Difference**: How differently models embed the same text
- **Neighbor Consistency**: Agreement on similar texts
- **Embedding Statistics**: Norms, dimensions, similarity patterns

## Configuration Options

```python
comparator = EmbeddingModelComparator(
    model1_name="Custom-Model-1",  # Your model names
    model2_name="Custom-Model-2"
)

results = comparator.run_comprehensive_evaluation(
    texts=texts,
    embeddings1=embeddings1,
    embeddings2=embeddings2,
    contexts=contexts,                    # Optional
    expected_outputs=expected_outputs,    # Optional
    create_csv=True,                     # Generate CSV reports
    output_dir="my_results"              # Output directory
)
```

## Example Use Cases

### 1. Comparing Different Model Versions
```python
# Compare BERT base vs large
comparator = EmbeddingModelComparator("BERT-Base", "BERT-Large")
```

### 2. Comparing Different Model Types
```python
# Compare sentence transformers vs OpenAI
comparator = EmbeddingModelComparator("SentenceTransformer", "OpenAI-Ada-002")
```

### 3. Comparing Fine-tuned vs Pre-trained
```python
# Compare before/after fine-tuning
comparator = EmbeddingModelComparator("Pre-trained", "Fine-tuned")
```

## Tips for Best Results

1. **Use Diverse Text Data**: Include various text types and lengths
2. **Provide Contexts**: Improves contextual relevancy metrics
3. **Include Expected Outputs**: Enables faithfulness evaluation
4. **Match Sample Counts**: Ensure both embedding arrays have same number of samples
5. **Quality Text Data**: Use real-world text that represents your use case

## Troubleshooting

### Common Issues

1. **Shape Mismatch Error**:
   ```python
   # Ensure same number of samples
   assert len(texts) == embeddings1.shape[0] == embeddings2.shape[0]
   ```

2. **Memory Issues with Large Datasets**:
   ```python
   # Process in batches for large datasets
   batch_size = 1000
   # Split data into batches and process separately
   ```

3. **Missing Dependencies**:
   ```bash
   pip install deepeval pandas numpy scikit-learn
   ```

## Advanced Usage

### Custom Metrics
```python
# Add custom metrics to the evaluation
from deepeval.metrics import HallucinationMetric

# In the run_comprehensive_evaluation method, add to metrics list:
metrics.append(HallucinationMetric(threshold=0.5))
```

### Batch Processing
```python
# For large datasets, process in batches
def process_large_dataset(texts, embeddings1, embeddings2, batch_size=1000):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_emb1 = embeddings1[i:i+batch_size]
        batch_emb2 = embeddings2[i:i+batch_size]
        
        # Process batch
        batch_results = comparator.run_comprehensive_evaluation(
            batch_texts, batch_emb1, batch_emb2
        )
        results.append(batch_results)
    
    return results
```

## Contributing

Feel free to extend this tool with additional metrics or functionality. The modular design makes it easy to add new evaluation methods or output formats.

## License

This tool is provided as-is for evaluation purposes. Please ensure you comply with the licenses of the dependencies (DeepEval, etc.) when using in production.