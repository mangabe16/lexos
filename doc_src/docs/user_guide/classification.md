# Classifying documents

**Classification** is a machine learning process used to determine the category, group, or authorship of text documents based on their underlying textual features. It operates under a clear lifecycle workflow: preprocessing the input data, initializing a desired machine learning model, training that model on known data, and evaluating its overall performance.

The `Classification` module equips users with a comprehensive, object-oriented framework to execute text classification experiments through two core components:

* **`Classifier` Context:** This class dictates the flow of every classifier desired, regardless of which model it will follow, ensuring all instances follow the same flow of preprocessing its own data, initializing the corresponding model, training it, and evaluate its results


* **`Pipeline` Strategies:** This class provides the necessary instructions to a 'Classifier' instance so it knows exactly how raw inputs (whether lists of text strings or pre-computed `CorpusStats` feature DataFrames) are scrubbed, tokenized, normalized, and mapped into the underlying estimator.

## Generating a Pipeline Strategy

To create a strategy, you can import an existing one from the classification module, or, as we shall learn later on, even create your own. In this case, we will create a Multi Layer Perceptron neural network. For this specific Pipeline, we must provide certain settings:

* `seed`: The seed to use for reproducibility
* `min_df`: The minimal document frequency for a term
* `test_size`: The size of the train/test split
* `cv_splits`: How many cross validation folds to run
* `include_bigrams`: Whether to use bigrams as a input feature #NOTE this is bound to change
* `use_smote`: Whether to apply SMOTE
* `run_holdout`: Whether to run a holdout validation step
* `run_cv`: Whether to run a cross validation step
* `mlp_kwargs`: The settings for the MLP model

Here is a sample setup

```python
# Import the MLPPipeline class
from lexos.classification import MLPPipeline

my_pipeline = MLPPipeline(
    seed=SEED,
    min_df=2,
    test_size=0.2,
    cv_splits=5,
    include_bigrams=True,
    use_smote=True,
    run_holdout=True,
    run_cv=True,
    mlp_kwargs={
        "hidden_layer_sizes": (64,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_iter": 1000,
    },
)
```

Notice how we haven't yet provide what is the data, but simply how the Classifier is to perform the classification and all the underlying necessary steps

#### How to write a new Pipeline
