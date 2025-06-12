import numpy as np
import pandas as pd
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from typing import List, Dict, Any
import json
import os
from datetime import datetime

class EmbeddingModelComparator:
    def __init__(self, model1_name: str = "Model_1", model2_name: str = "Model_2"):
        self.model1_name = model1_name
        self.model2_name = model2_name
        
    def calculate_embedding_metrics(self, embeddings1: np.ndarray, embeddings2: np.ndarray, 
                                  texts: List[str]) -> Dict[str, Any]:
        """Calculate various embedding-specific metrics"""
        
        # Cosine similarity between embeddings
        cos_sim_matrix1 = cosine_similarity(embeddings1)
        cos_sim_matrix2 = cosine_similarity(embeddings2)
        
        # Average cosine similarity difference
        cos_sim_diff = np.mean(np.abs(cos_sim_matrix1 - cos_sim_matrix2))
        
        # Embedding space statistics
        model1_stats = {
            'mean_norm': np.mean(np.linalg.norm(embeddings1, axis=1)),
            'std_norm': np.std(np.linalg.norm(embeddings1, axis=1)),
            'dimensionality': embeddings1.shape[1],
            'mean_cosine_sim': np.mean(cos_sim_matrix1[np.triu_indices_from(cos_sim_matrix1, k=1)])
        }
        
        model2_stats = {
            'mean_norm': np.mean(np.linalg.norm(embeddings2, axis=1)),
            'std_norm': np.std(np.linalg.norm(embeddings2, axis=1)),
            'dimensionality': embeddings2.shape[1],
            'mean_cosine_sim': np.mean(cos_sim_matrix2[np.triu_indices_from(cos_sim_matrix2, k=1)])
        }
        
        # Clustering consistency (using embeddings to find nearest neighbors)
        def get_top_k_neighbors(embeddings, k=5):
            similarities = cosine_similarity(embeddings)
            top_k_indices = []
            for i in range(len(embeddings)):
                # Get indices of top-k most similar (excluding self)
                similar_indices = np.argsort(similarities[i])[::-1][1:k+1]
                top_k_indices.append(similar_indices.tolist())
            return top_k_indices
        
        neighbors1 = get_top_k_neighbors(embeddings1)
        neighbors2 = get_top_k_neighbors(embeddings2)
        
        # Calculate neighbor overlap
        neighbor_overlap = []
        for n1, n2 in zip(neighbors1, neighbors2):
            overlap = len(set(n1) & set(n2)) / len(set(n1) | set(n2))
            neighbor_overlap.append(overlap)
        
        return {
            'cosine_similarity_difference': cos_sim_diff,
            'model1_stats': model1_stats,
            'model2_stats': model2_stats,
            'average_neighbor_overlap': np.mean(neighbor_overlap),
            'neighbor_overlap_std': np.std(neighbor_overlap)
        }
    
    def create_test_cases_for_deepeval(self, texts: List[str], 
                                     embeddings1: np.ndarray, 
                                     embeddings2: np.ndarray,
                                     contexts: List[str] = None,
                                     expected_outputs: List[str] = None) -> tuple:
        """Create test cases for both models to use with deepeval metrics"""
        
        test_cases_model1 = []
        test_cases_model2 = []
        
        for i, text in enumerate(texts):
            # For deepeval, we need to simulate responses based on embeddings
            # This is a creative approach since deepeval expects text responses
            
            # Use embedding similarity to create pseudo-responses
            if embeddings1.shape[0] > 1:
                # Find most similar text based on embeddings
                similarities1 = cosine_similarity([embeddings1[i]], embeddings1)[0]
                similarities2 = cosine_similarity([embeddings2[i]], embeddings2)[0]
                
                # Get top similar indices (excluding self)
                top_sim1 = np.argsort(similarities1)[::-1][1:4]  # Top 3 similar
                top_sim2 = np.argsort(similarities2)[::-1][1:4]
                
                # Create pseudo-responses based on similarity
                response1 = f"Similar to: {', '.join([texts[idx] for idx in top_sim1])}"
                response2 = f"Similar to: {', '.join([texts[idx] for idx in top_sim2])}"
            else:
                response1 = text
                response2 = text
            
            # Create test cases
            context = contexts[i] if contexts else text
            expected = expected_outputs[i] if expected_outputs else text
            
            test_case1 = LLMTestCase(
                input=text,
                actual_output=response1,
                expected_output=expected,
                context=[context]
            )
            
            test_case2 = LLMTestCase(
                input=text,
                actual_output=response2,
                expected_output=expected,
                context=[context]
            )
            
            test_cases_model1.append(test_case1)
            test_cases_model2.append(test_case2)
        
        return test_cases_model1, test_cases_model2
    
    def run_comprehensive_evaluation(self, 
                                   texts: List[str],
                                   embeddings1: np.ndarray,
                                   embeddings2: np.ndarray,
                                   contexts: List[str] = None,
                                   expected_outputs: List[str] = None,
                                   create_csv: bool = True,
                                   output_dir: str = "embedding_comparison_results") -> Dict[str, Any]:
        """Run comprehensive evaluation comparing both models"""
        
        print("Calculating embedding-specific metrics...")
        embedding_metrics = self.calculate_embedding_metrics(embeddings1, embeddings2, texts)
        
        print("Creating test cases for deepeval...")
        test_cases1, test_cases2 = self.create_test_cases_for_deepeval(
            texts, embeddings1, embeddings2, contexts, expected_outputs
        )
        
        # Initialize metrics
        metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            ContextualRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
            BiasMetric(threshold=0.5),
            ToxicityMetric(threshold=0.5)
        ]
        
        print(f"Evaluating {self.model1_name}...")
        dataset1 = EvaluationDataset(test_cases=test_cases1)
        results1 = evaluate(dataset1, metrics)
        
        print(f"Evaluating {self.model2_name}...")
        dataset2 = EvaluationDataset(test_cases=test_cases2)
        results2 = evaluate(dataset2, metrics)
        
        # Compile results
        comparison_results = {
            'embedding_metrics': embedding_metrics,
            'deepeval_results': {
                self.model1_name: self._extract_metric_scores(results1),
                self.model2_name: self._extract_metric_scores(results2)
            },
            'summary': self._create_summary(embedding_metrics, results1, results2)
        }
        
        # Create CSV reports
        if create_csv:
            self.create_csv_reports(comparison_results, output_dir)
        
        return comparison_results
    
    def _extract_metric_scores(self, results) -> Dict[str, float]:
        """Extract scores from deepeval results"""
        scores = {}
        for result in results:
            for metric_result in result.metrics_data:
                metric_name = metric_result.name
                scores[metric_name] = metric_result.score
        return scores
    
    def _create_summary(self, embedding_metrics: Dict, results1, results2) -> Dict[str, Any]:
        """Create a summary comparison"""
        
        scores1 = self._extract_metric_scores(results1)
        scores2 = self._extract_metric_scores(results2)
        
        # Calculate which model performs better on each metric
        better_model = {}
        for metric in scores1.keys():
            if metric in scores2:
                if scores1[metric] > scores2[metric]:
                    better_model[metric] = self.model1_name
                elif scores2[metric] > scores1[metric]:
                    better_model[metric] = self.model2_name
                else:
                    better_model[metric] = "Tie"
        
        return {
            'embedding_space_comparison': {
                'cosine_similarity_difference': embedding_metrics['cosine_similarity_difference'],
                'neighbor_consistency': embedding_metrics['average_neighbor_overlap'],
                'dimensionality_match': embedding_metrics['model1_stats']['dimensionality'] == 
                                      embedding_metrics['model2_stats']['dimensionality']
            },
            'deepeval_winner_by_metric': better_model,
            'overall_winner': max(better_model.values(), key=list(better_model.values()).count)
        }
    
    def create_csv_reports(self, results: Dict[str, Any], output_dir: str = "embedding_comparison_results") -> None:
        """Create comprehensive CSV reports from evaluation results"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Overall Model Comparison CSV
        self._create_model_comparison_csv(results, output_dir, timestamp)
        
        # 2. Detailed Metrics CSV
        self._create_detailed_metrics_csv(results, output_dir, timestamp)
        
        # 3. Embedding Statistics CSV
        self._create_embedding_stats_csv(results, output_dir, timestamp)
        
        # 4. Per-Sample Results CSV (if available)
        self._create_per_sample_csv(results, output_dir, timestamp)
        
        print(f"CSV reports created in directory: {output_dir}")
    
    def _create_model_comparison_csv(self, results: Dict[str, Any], output_dir: str, timestamp: str):
        """Create high-level model comparison CSV"""
        
        comparison_data = []
        deepeval_results = results['deepeval_results']
        
        # Get all unique metrics
        all_metrics = set()
        for model_results in deepeval_results.values():
            all_metrics.update(model_results.keys())
        
        # Create comparison rows
        for model_name, model_results in deepeval_results.items():
            row = {
                'model_name': model_name,
                'evaluation_timestamp': timestamp,
                'embedding_dimension': results['embedding_metrics']['model1_stats']['dimensionality'] 
                                     if model_name == self.model1_name 
                                     else results['embedding_metrics']['model2_stats']['dimensionality'],
                'cosine_similarity_difference': results['embedding_metrics']['cosine_similarity_difference'],
                'neighbor_overlap_score': results['embedding_metrics']['average_neighbor_overlap']
            }
            
            # Add all deepeval metrics
            for metric in all_metrics:
                row[f'{metric}_score'] = model_results.get(metric, np.nan)
            
            # Add embedding stats
            model_stats_key = 'model1_stats' if model_name == self.model1_name else 'model2_stats'
            embedding_stats = results['embedding_metrics'][model_stats_key]
            
            row.update({
                'mean_embedding_norm': embedding_stats['mean_norm'],
                'std_embedding_norm': embedding_stats['std_norm'],
                'mean_cosine_similarity': embedding_stats['mean_cosine_sim']
            })
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(f"{output_dir}/model_comparison_{timestamp}.csv", index=False)
        print(f"Created: model_comparison_{timestamp}.csv")
    
    def _create_detailed_metrics_csv(self, results: Dict[str, Any], output_dir: str, timestamp: str):
        """Create detailed metrics breakdown CSV"""
        
        detailed_data = []
        summary = results['summary']
        
        # Get all metrics and their winners
        for metric, winner in summary['deepeval_winner_by_metric'].items():
            model1_score = results['deepeval_results'][self.model1_name].get(metric, np.nan)
            model2_score = results['deepeval_results'][self.model2_name].get(metric, np.nan)
            
            detailed_data.append({
                'metric_name': metric,
                'metric_category': self._categorize_metric(metric),
                f'{self.model1_name}_score': model1_score,
                f'{self.model2_name}_score': model2_score,
                'score_difference': abs(model1_score - model2_score) if not (np.isnan(model1_score) or np.isnan(model2_score)) else np.nan,
                'winner': winner,
                'evaluation_timestamp': timestamp
            })
        
        df = pd.DataFrame(detailed_data)
        df.to_csv(f"{output_dir}/detailed_metrics_{timestamp}.csv", index=False)
        print(f"Created: detailed_metrics_{timestamp}.csv")
    
    def _create_embedding_stats_csv(self, results: Dict[str, Any], output_dir: str, timestamp: str):
        """Create embedding statistics CSV"""
        
        embedding_data = []
        embedding_metrics = results['embedding_metrics']
        
        # Model 1 stats
        stats1 = embedding_metrics['model1_stats']
        embedding_data.append({
            'model_name': self.model1_name,
            'dimensionality': stats1['dimensionality'],
            'mean_norm': stats1['mean_norm'],
            'std_norm': stats1['std_norm'],
            'mean_cosine_similarity': stats1['mean_cosine_sim'],
            'evaluation_timestamp': timestamp
        })
        
        # Model 2 stats
        stats2 = embedding_metrics['model2_stats']
        embedding_data.append({
            'model_name': self.model2_name,
            'dimensionality': stats2['dimensionality'],
            'mean_norm': stats2['mean_norm'],
            'std_norm': stats2['std_norm'],
            'mean_cosine_similarity': stats2['mean_cosine_sim'],
            'evaluation_timestamp': timestamp
        })
        
        # Add comparison metrics
        comparison_row = {
            'model_name': 'COMPARISON',
            'cosine_similarity_difference': embedding_metrics['cosine_similarity_difference'],
            'average_neighbor_overlap': embedding_metrics['average_neighbor_overlap'],
            'neighbor_overlap_std': embedding_metrics['neighbor_overlap_std'],
            'dimensionality_match': stats1['dimensionality'] == stats2['dimensionality'],
            'evaluation_timestamp': timestamp
        }
        embedding_data.append(comparison_row)
        
        df = pd.DataFrame(embedding_data)
        df.to_csv(f"{output_dir}/embedding_statistics_{timestamp}.csv", index=False)
        print(f"Created: embedding_statistics_{timestamp}.csv")
    
    def _create_per_sample_csv(self, results: Dict[str, Any], output_dir: str, timestamp: str):
        """Create per-sample results CSV if test case data is available"""
        
        # This would require storing per-sample results during evaluation
        # For now, create a template that can be extended
        sample_data = {
            'sample_id': ['sample_1', 'sample_2', 'sample_3'],
            'input_text': ['Example input 1', 'Example input 2', 'Example input 3'],
            f'{self.model1_name}_embedding_norm': [1.0, 1.1, 0.9],
            f'{self.model2_name}_embedding_norm': [1.2, 0.8, 1.0],
            'cosine_similarity_between_models': [0.85, 0.92, 0.78],
            'evaluation_timestamp': [timestamp] * 3
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(f"{output_dir}/per_sample_results_{timestamp}.csv", index=False)
        print(f"Created: per_sample_results_{timestamp}.csv (template)")
    
    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize metrics for better organization"""
        if 'relevancy' in metric_name.lower() or 'relevance' in metric_name.lower():
            return 'Relevancy'
        elif 'faithfulness' in metric_name.lower():
            return 'Faithfulness'
        elif 'precision' in metric_name.lower() or 'recall' in metric_name.lower():
            return 'Precision/Recall'
        elif 'bias' in metric_name.lower():
            return 'Bias'
        elif 'toxicity' in metric_name.lower():
            return 'Safety'
        elif 'hallucination' in metric_name.lower():
            return 'Accuracy'
        else:
            return 'Other'

# Data Loading Helper Functions
def load_embeddings_from_csv(filepath: str, embedding_columns: List[str] = None) -> np.ndarray:
    """
    Load embeddings from CSV file
    
    Args:
        filepath: Path to CSV file
        embedding_columns: List of column names containing embedding values
                          If None, assumes all numeric columns are embeddings
    
    Returns:
        numpy array of embeddings
    """
    df = pd.read_csv(filepath)
    
    if embedding_columns:
        embeddings = df[embedding_columns].values
    else:
        # Auto-detect numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        embeddings = df[numeric_cols].values
    
    return embeddings.astype(float)

def load_embeddings_from_npy(filepath: str) -> np.ndarray:
    """Load embeddings from .npy file"""
    return np.load(filepath)

def load_embeddings_from_json(filepath: str, embedding_key: str = 'embeddings') -> np.ndarray:
    """
    Load embeddings from JSON file
    
    Args:
        filepath: Path to JSON file
        embedding_key: Key in JSON containing embeddings list
    
    Returns:
        numpy array of embeddings
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        embeddings = np.array(data)
    elif isinstance(data, dict) and embedding_key in data:
        embeddings = np.array(data[embedding_key])
    else:
        raise ValueError(f"Could not find embeddings in JSON file. Expected key: {embedding_key}")
    
    return embeddings.astype(float)

def create_sample_data_files():
    """Create sample data files to demonstrate expected input formats"""
    
    # Sample texts
    texts = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks",
        "What are transformers in AI?",
        "How to train a model?"
    ]
    
    # Sample embeddings (different dimensions to show comparison)
    embeddings1 = np.random.rand(5, 768)  # BERT-like
    embeddings2 = np.random.rand(5, 1536)  # OpenAI-like
    
    # Sample contexts
    contexts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses multiple layers of neural networks",
        "Neural networks are inspired by biological neural networks",
        "Transformers use attention mechanisms for processing",
        "Model training involves optimizing parameters"
    ]
    
    # Create CSV format (embeddings as separate columns)
    df1 = pd.DataFrame(embeddings1)
    df1['text'] = texts
    df1['context'] = contexts
    df1.to_csv('sample_embeddings_model1.csv', index=False)
    
    df2 = pd.DataFrame(embeddings2)
    df2['text'] = texts  
    df2['context'] = contexts
    df2.to_csv('sample_embeddings_model2.csv', index=False)
    
    # Create numpy format
    np.save('sample_embeddings_model1.npy', embeddings1)
    np.save('sample_embeddings_model2.npy', embeddings2)
    
    # Create JSON format
    json_data1 = {
        'embeddings': embeddings1.tolist(),
        'texts': texts,
        'contexts': contexts,
        'model_name': 'BERT',
        'embedding_dim': 768
    }
    
    json_data2 = {
        'embeddings': embeddings2.tolist(),
        'texts': texts,
        'contexts': contexts,
        'model_name': 'OpenAI',
        'embedding_dim': 1536
    }
    
    with open('sample_embeddings_model1.json', 'w') as f:
        json.dump(json_data1, f, indent=2)
        
    with open('sample_embeddings_model2.json', 'w') as f:
        json.dump(json_data2, f, indent=2)
    
    # Create texts file
    pd.DataFrame({
        'text': texts,
        'context': contexts,
        'expected_output': [f"Answer about {topic}" for topic in ["ML", "DL", "NN", "Transformers", "Training"]]
    }).to_csv('sample_texts.csv', index=False)
    
    print("Sample data files created:")
    print("- sample_embeddings_model1.csv/npy/json")
    print("- sample_embeddings_model2.csv/npy/json") 
    print("- sample_texts.csv")

# Example usage with different data formats
def example_usage_with_files():
    """Example showing how to use the comparator with different file formats"""
    
    # Create sample data files first
    create_sample_data_files()
    
    # Method 1: Load from CSV files
    print("Loading from CSV files...")
    embeddings1 = load_embeddings_from_csv('sample_embeddings_model1.csv', 
                                          embedding_columns=[str(i) for i in range(768)])
    embeddings2 = load_embeddings_from_csv('sample_embeddings_model2.csv',
                                          embedding_columns=[str(i) for i in range(1536)])
    
    # Load text data
    text_df = pd.read_csv('sample_texts.csv')
    texts = text_df['text'].tolist()
    contexts = text_df['context'].tolist()
    expected_outputs = text_df['expected_output'].tolist()
    
    # Method 2: Load from numpy files
    # embeddings1 = load_embeddings_from_npy('sample_embeddings_model1.npy')
    # embeddings2 = load_embeddings_from_npy('sample_embeddings_model2.npy')
    
    # Method 3: Load from JSON files
    # embeddings1 = load_embeddings_from_json('sample_embeddings_model1.json')
    # embeddings2 = load_embeddings_from_json('sample_embeddings_model2.json')
    
    # Run comparison
    comparator = EmbeddingModelComparator("BERT-Base", "OpenAI-Ada-002")
    
    results = comparator.run_comprehensive_evaluation(
        texts=texts,
        embeddings1=embeddings1,
        embeddings2=embeddings2,
        contexts=contexts,
        expected_outputs=expected_outputs,
        create_csv=True,
        output_dir="my_embedding_comparison"
    )
    
    print(f"\nComparison complete! Check the 'my_embedding_comparison' directory for CSV results.")
    print(f"Overall winner: {results['summary']['overall_winner']}")

# Expected Input Data Formats
"""
EXPECTED INPUT DATA FORMATS:

1. EMBEDDINGS:
   - Format: numpy array with shape (n_samples, embedding_dimension)
   - Each row is one embedding vector
   - Can have different dimensions for different models
   
   Example:
   model1_embeddings.shape = (100, 768)    # 100 samples, 768-dim embeddings
   model2_embeddings.shape = (100, 1536)   # same 100 samples, 1536-dim embeddings

2. TEXTS (Required):
   - Format: List of strings
   - The original text that was embedded
   - Must match the number of embeddings
   
   Example:
   texts = ["What is AI?", "How does ML work?", ...]

3. CONTEXTS (Optional but recommended):
   - Format: List of strings  
   - Additional context for each text
   - Used for contextual relevancy metrics
   
   Example:
   contexts = ["AI context...", "ML context...", ...]

4. EXPECTED_OUTPUTS (Optional):
   - Format: List of strings
   - Expected/reference outputs for evaluation
   - Used for faithfulness and relevancy metrics
   
   Example:
   expected_outputs = ["AI is...", "ML works by...", ...]

FILE FORMATS SUPPORTED:
- CSV: Embeddings as columns, with text/context columns
- NumPy (.npy): Direct embedding arrays  
- JSON: Structured format with embeddings, texts, contexts

CSV COLUMNS STRUCTURE:
- Embedding columns: "0", "1", "2", ... (numeric column names)
- Text column: "text" 
- Context column: "context"
- Expected output column: "expected_output"
"""
def example_usage():
    """Basic example with generated data"""
    # Sample data - replace with your actual data
    texts = [
        "What is machine learning?",
        "How does deep learning work?", 
        "Explain neural networks",
        "What are transformers in AI?",
        "How to train a model?"
    ]
    
    # Sample embeddings - replace with your actual embeddings
    embeddings_model1 = np.random.rand(5, 768)  # Example: BERT-like embeddings
    embeddings_model2 = np.random.rand(5, 1536)  # Example: OpenAI embeddings
    
    # Optional: contexts and expected outputs
    contexts = [
        "Machine learning context...",
        "Deep learning context...",
        "Neural network context...", 
        "Transformer context...",
        "Model training context..."
    ]
    
    expected_outputs = [
        "ML is a subset of AI...",
        "Deep learning uses neural networks...",
        "Neural networks are inspired by brain...",
        "Transformers use attention mechanism...",
        "Training involves optimization..."
    ]
    
    # Initialize comparator
    comparator = EmbeddingModelComparator("BERT", "OpenAI-ADA")
    
    # Run evaluation with CSV output
    results = comparator.run_comprehensive_evaluation(
        texts=texts,
        embeddings1=embeddings_model1,
        embeddings2=embeddings_model2,
        contexts=contexts,
        expected_outputs=expected_outputs,
        create_csv=True,
        output_dir="embedding_results"
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EMBEDDING MODEL COMPARISON SUMMARY")
    print("="*50)
    print(f"Cosine Similarity Difference: {results['embedding_metrics']['cosine_similarity_difference']:.4f}")
    print(f"Average Neighbor Overlap: {results['embedding_metrics']['average_neighbor_overlap']:.4f}")
    print(f"Overall Winner: {results['summary']['overall_winner']}")
    
    print("\nWinner by Metric:")
    for metric, winner in results['summary']['deepeval_winner_by_metric'].items():
        print(f"  {metric}: {winner}")

if __name__ == "__main__":
    # Run example with file loading
    example_usage_with_files()
    
    # Run basic example  
    # example_usage()