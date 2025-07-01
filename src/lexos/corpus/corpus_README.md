# Lexos Corpus Module: Centralized Document Management and Analysis

The `lexos.corpus` module serves as the core foundation for document management and statistical analysis in the Lexos ecosystem. It provides centralized storage, metadata management, and inter-module communication capabilities that enable seamless integration with analysis modules like clustering, topic modeling, classification, and content analysis.

---

## Overview for Developers

The Corpus module acts as a **central hub** for document management across the Lexos ecosystem. Other module developers should understand how to:

1. **Attach analysis results** to documents using metadata strategies
2. **Manage statistical boundaries** and corpus state validation
3. **Implement communication architecture** for result sharing between modules
4. **Handle spaCy document serialization** robustly for cross-module compatibility

---

## Core Classes

### `Corpus` (`corpus.py`)
The main container for managing collections of documents. Provides document storage, metadata management, and inter-module communication capabilities.

**Key Features:**
- **Document Storage**: File-based storage with msgpack serialization for portability
- **Metadata Management**: Comprehensive metadata handling at corpus and document levels
- **Statistical Analysis**: Integration with `CorpusStats` for advanced corpus analysis
- **Communication Architecture**: Result sharing and validation between analysis modules
- **Serialization Robustness**: Enhanced spaCy document handling with automatic fallback mechanisms

**Inter-Module Integration Points:**
```python
# For other module developers - attach analysis results
corpus.import_analysis_results('kmeans', {
    'clusters': cluster_assignments,
    'silhouette_score': 0.75,
    'parameters': {'n_clusters': 5, 'algorithm': 'kmeans++'}
}, version='1.2.0')

# Validate compatibility before analysis
compatibility = corpus.validate_analysis_compatibility('kmeans')
if not compatibility['compatible']:
    print(f"Warning: {compatibility['reason']}")
```

### `Record` (`record.py`)
Individual document container with robust metadata and serialization capabilities.

**Key Features:**
- **Content Storage**: Supports raw text and spaCy `Doc` objects
- **Metadata Management**: Flexible `meta` attribute for arbitrary data storage
- **Serialization**: msgpack-based with SHA256 integrity verification
- **spaCy Integration**: Robust handling of linguistic annotations and custom extensions

**Metadata Attachment Strategies for Module Developers:**
```python
# Strategy 1: Use Record.meta for module-specific results
record.meta['classification'] = {
    'predicted_class': 'positive',
    'confidence': 0.89,
    'model_version': '2.1.0'
}

# Strategy 2: Use spaCy Doc.user_data for linguistic annotations
if record.is_parsed:
    record.content.user_data['sentiment_scores'] = {
        'positive': 0.8, 'negative': 0.2, 'neutral': 0.0
    }

# Strategy 3: Use spaCy Doc.cats for classification results
if record.is_parsed:
    record.content.cats = {
        'POSITIVE': 0.89, 'NEGATIVE': 0.11
    }
```

### `CorpusStats` (`corpus_stats.py`)
Advanced statistical analysis engine with **cached IQR properties** for performance optimization.

**Key Features:**
- **Cached Properties**: IQR calculations cached for improved performance (new in v1.5)
- **Statistical Boundaries**: Clear boundaries between corpus-level and document-level statistics
- **Advanced Metrics**: Lexical diversity, Zipf analysis, quality metrics, group comparisons
- **Visualization**: Integrated plotting capabilities for statistical distributions

**Performance Improvements (Recent Updates):**
```python
# Cached IQR properties eliminate redundant calculations
stats = corpus.get_stats()

# First access calculates and caches
iqr_values = stats.iqr_values  # (Q1, Q3, IQR)
iqr_bounds = stats.iqr_bounds  # (lower_bound, upper_bound)
outliers = stats.iqr_outliers  # List of outlier documents

# Subsequent accesses use cached values (significant speed improvement)
outliers_again = stats.get_iqr_outliers()  # Uses cached IQR calculations
```

### `LexosModelCache` and `RecordsDict` (`utils.py`)
Utility classes for efficient model management and type-safe record storage.

---

## Metadata Management Strategies for Module Developers

### When to Use Each Metadata Approach

| Metadata Location | Use Case | Example Data | Persistence |
|------------------|----------|--------------|-------------|
| **`Record.meta`** | Module results, configuration, arbitrary data | Classification results, analysis parameters | ✅ Serialized with record |
| **`Doc.user_data`** | Linguistic annotations, document-level scores | Sentiment scores, topic distributions | ✅ Serialized with spaCy Doc |
| **`Doc.cats`** | Classification categories (spaCy standard) | Text classification probabilities | ✅ Serialized with spaCy Doc |
| **Custom Extensions** | Token-level annotations, linguistic features | Named entities, dependency features | ⚠️ Requires extension registration |

### Recommended Patterns for Module Integration

#### Pattern 1: Store Analysis Configuration and Results
```python
# When your module processes a corpus
def analyze_corpus(corpus, **params):
    # Store analysis parameters
    analysis_config = {
        'module': 'your_module',
        'version': '1.0.0',
        'parameters': params,
        'timestamp': datetime.now().isoformat()
    }
    
    # Process each document
    results = []
    for record_id, record in corpus.records.items():
        # Your analysis logic here
        result = your_analysis_function(record.content)
        
        # Store result in record metadata
        record.meta['your_module'] = {
            'result': result,
            'config': analysis_config
        }
        results.append(result)
    
    # Store corpus-level results
    corpus.import_analysis_results('your_module', {
        'individual_results': results,
        'summary_statistics': calculate_summary(results),
        'configuration': analysis_config
    })
    
    return results
```

#### Pattern 2: Handle spaCy Document Annotations
```python
# When working with linguistic annotations
def add_linguistic_features(record):
    if not record.is_parsed:
        print(f"Warning: {record.name} not parsed with spaCy")
        return
    
    doc = record.content
    
    # Add document-level scores to user_data
    doc.user_data['linguistic_complexity'] = calculate_complexity(doc)
    doc.user_data['readability_score'] = calculate_readability(doc)
    
    # Add classification scores to cats
    doc.cats = {
        'FORMAL': 0.7,
        'INFORMAL': 0.3
    }
    
    # Note: Changes are automatically saved when record is serialized
```

#### Pattern 3: Validate Corpus State Before Analysis
```python
# When your module depends on corpus state
def safe_analysis(corpus, module_name):
    # Check if previous analysis exists and is still valid
    if module_name in corpus.analysis_results:
        compatibility = corpus.validate_analysis_compatibility(module_name)
        
        if compatibility['compatible']:
            print("Using cached analysis results")
            return corpus.get_analysis_results(module_name)
        else:
            print(f"Corpus changed: {compatibility['reason']}")
            print(f"Recommendation: {compatibility['recommendation']}")
    
    # Proceed with fresh analysis
    return run_new_analysis(corpus)
```

---

## Communication Architecture

### Inter-Module Result Sharing

The Corpus module implements a **communication architecture** that enables modules to share results and validate compatibility:

```python
# Module A stores results
corpus.import_analysis_results('topic_modeling', {
    'topics': topic_assignments,
    'model_params': {'n_topics': 10, 'alpha': 0.1}
})

# Module B retrieves and uses results
try:
    topic_results = corpus.get_analysis_results('topic_modeling')
    topics = topic_results['results']['topics']
    
    # Use topic assignments for further analysis
    cluster_with_topics(corpus, topics)
except ValueError:
    print("Topic modeling results not available")
```

### Corpus State Validation

The module tracks corpus state to detect when cached results become invalid:

```python
# Automatic validation when corpus changes
initial_fingerprint = corpus._generate_corpus_fingerprint()

# ... add/remove documents ...

# Check if previous analysis is still valid
compatibility = corpus.validate_analysis_compatibility('clustering')
if not compatibility['compatible']:
    # Re-run analysis because corpus state changed
    run_clustering_analysis(corpus)
```



## Installation and Dependencies

```bash
# Core dependencies
pip install spacy pandas numpy matplotlib seaborn plotly scipy pydantic

# spaCy language model
python -m spacy download en_core_web_sm

# For development/testing
pip install pytest pytest-cov
```

---

## Integration Examples

### Example 1: Classification Module Integration
```python
from lexos.corpus import Corpus
import spacy

# Setup
nlp = spacy.load("en_core_web_sm")
corpus = Corpus(name="classification_corpus")

# Add documents
texts = ["Positive example text", "Negative example text"]
for i, text in enumerate(texts):
    doc = nlp(text)
    corpus.add(doc, name=f"doc_{i}", model="en_core_web_sm")

# Classification module workflow
def classify_documents(corpus, model):
    results = []
    
    for record_id, record in corpus.records.items():
        if record.is_parsed:
            # Classify document
            prediction = model.predict(record.content.text)
            confidence = model.predict_proba(record.content.text).max()
            
            # Store in spaCy cats (standard approach)
            record.content.cats = {
                'POSITIVE': confidence if prediction == 1 else 1-confidence,
                'NEGATIVE': confidence if prediction == 0 else 1-confidence
            }
            
            # Store detailed results in record metadata
            record.meta['classification'] = {
                'predicted_class': 'positive' if prediction == 1 else 'negative',
                'confidence': float(confidence),
                'model_name': model.__class__.__name__,
                'features_used': 'text_content'
            }
            
            results.append({
                'record_id': record_id,
                'prediction': prediction,
                'confidence': confidence
            })
    
    # Store corpus-level results
    corpus.import_analysis_results('text_classification', {
        'predictions': results,
        'model_info': {
            'algorithm': model.__class__.__name__,
            'training_data': 'external',
            'accuracy': 0.95  # If known
        },
        'summary': {
            'positive_count': sum(1 for r in results if r['prediction'] == 1),
            'negative_count': sum(1 for r in results if r['prediction'] == 0),
            'avg_confidence': sum(r['confidence'] for r in results) / len(results)
        }
    })
    
    return results
```

### Example 2: Topic Modeling Integration
```python
def topic_modeling_integration(corpus, n_topics=5):
    # Get corpus statistics for analysis
    stats = corpus.get_stats()
    
    # Check corpus quality
    quality_metrics = stats.corpus_quality_metrics
    if quality_metrics['vocabulary_richness']['sampling_adequacy'] == 'insufficient':
        print("Warning: Corpus may be too small for reliable topic modeling")
    
    # Run topic modeling (pseudocode)
    topic_model = fit_topic_model(corpus, n_topics=n_topics)
    
    # Store results in documents
    for record_id, record in corpus.records.items():
        if record.is_parsed:
            # Get topic distribution for document
            topic_dist = topic_model.transform([record.content.text])[0]
            
            # Store in user_data
            record.content.user_data['topic_distribution'] = {
                f'topic_{i}': float(prob) for i, prob in enumerate(topic_dist)
            }
            
            # Store dominant topic in metadata
            dominant_topic = topic_dist.argmax()
            record.meta['topic_modeling'] = {
                'dominant_topic': int(dominant_topic),
                'topic_probability': float(topic_dist[dominant_topic]),
                'entropy': float(calculate_entropy(topic_dist))
            }
    
    # Store model-level results
    corpus.import_analysis_results('topic_modeling', {
        'model_type': 'LDA',
        'n_topics': n_topics,
        'topics': topic_model.get_topic_words(),
        'coherence_score': calculate_coherence(topic_model),
        'perplexity': topic_model.log_perplexity(corpus_texts)
    })
```

---

## Testing and Quality Assurance

The module maintains **94% test coverage** with comprehensive test suites:

```bash
# Run corpus module tests
uv run pytest tests/corpus/ --cov=src/lexos/corpus

# Run specific test categories
uv run pytest tests/corpus/test_corpus_stats.py  # Statistical functionality
uv run pytest tests/corpus/test_record.py       # Document serialization
uv run pytest tests/corpus/test_corpus.py       # Core corpus functionality
```

**Recent Testing Improvements:**
- **210+ passing tests** with comprehensive edge case coverage
- **Performance benchmarking** for cached properties
- **Serialization robustness testing** with various spaCy configurations
- **Inter-module compatibility testing** for communication architecture

---

## Development Guidelines for Module Integration

### Best Practices

1. **Always check `record.is_parsed`** before accessing spaCy-specific features
2. **Use appropriate metadata storage** based on data type and persistence needs
3. **Validate corpus state** before running expensive analyses
4. **Store analysis parameters** along with results for reproducibility
5. **Handle serialization gracefully** with fallback mechanisms

### Error Handling Patterns

```python
# Robust record processing
def process_record(record):
    try:
        if record.is_parsed:
            # spaCy-based processing
            result = analyze_spacy_doc(record.content)
        else:
            # Fallback to text processing
            result = analyze_raw_text(record.content)
        
        # Store result safely
        record.meta['your_module'] = {
            'result': result,
            'processing_type': 'spacy' if record.is_parsed else 'text',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        # Log error and store failure information
        record.meta['your_module'] = {
            'error': str(e),
            'failed_at': datetime.now().isoformat(),
            'processing_attempted': True
        }
        logger.error(f"Failed to process {record.name}: {e}")
```

### Performance Considerations

```python
# Efficient corpus processing
def efficient_corpus_analysis(corpus):
    # Get statistics once and reuse
    stats = corpus.get_stats()
    
    # Use cached properties
    outliers = stats.iqr_outliers  # Uses cached IQR calculations
    quality = stats.corpus_quality_metrics  # Uses cached document stats
    
    # Batch process documents
    for record_id, record in corpus.records.items():
        # Skip outliers if needed
        if any(record_id == outlier[0] for outlier in outliers):
            continue
        
        # Process record efficiently
        process_record_efficiently(record)
```

---

## Future Development

### Planned Enhancements
- **Database integration options** for large-scale deployments
- **Distributed corpus processing** capabilities
- **Enhanced inter-module APIs** with schema validation
- **Real-time collaboration features** for multi-user scenarios

### Contributing
For developers adding new analysis modules to Lexos:

1. **Review this README** for integration patterns
2. **Study existing module integrations** in the test suite
3. **Follow metadata storage conventions** outlined above
4. **Implement compatibility validation** for your module's results
5. **Add comprehensive tests** for corpus integration scenarios

The Corpus module is designed to be the foundation for all text analysis workflows in Lexos. By following these patterns, your module will integrate seamlessly with the broader ecosystem while maintaining robustness and performance.