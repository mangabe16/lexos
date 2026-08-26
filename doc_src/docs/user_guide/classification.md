# Classifying documents

## Overview

**Classification** is a predictrive modeling process in which machine learning models use different classification algorithms to predict the correct label for input data. In the context of NLP, it is used to determine the category, group, or authorship of text documents based on their underlying textual features.

The `lexos.classification` module equips users with a comprehensive, object-oriented framework to categorize and classify text documents using an uniform sequence of steps, whose implementation and execution are defined according to different available classifiers pipelines, allowing for an uniform flow to be executed by different desired stategies. This is possible through two core components:

* **`Classifier` Context:** This class dictates the flow of every classifier created via this module, regardless of which model it will follow, ensuring all instances follow the same flow of preprocessing its own data, initializing the corresponding model, training it, and evaluating its results.

* **`Pipeline` Strategies:** This class provides the necessary instructions to a 'Classifier' instance so it knows exactly how raw inputs (whether lists of text strings or pre-computed `CorpusStats` feature DataFrames) are scrubbed, tokenized, normalized, and mapped into the underlying estimator.


## Declaring a Pipeline Strategy

To declare a strategy, you can import an existing one from the classification module, or, as we shall learn later on, even create your own. In this case, we will create a Pipeline for a Decision Tree. For this specific Pipeline, after it being properly imported, we must provide the following settings:

* `seed`: The seed to use for reproducibility
* `min_df`: The minimal document frequency for a term
* `include_bigrams`: Whether to use bigrams as a input feature
* `tree_kwargs`: TO FINISH

Here is a sample setup:

```python
from lexos.classification import DecisionTreePipeline

my_pipeline = DecisionTreePipeline(
    seed=SEED, # Assuming a seed has already been defined
    min_df=2,
    include_bigrams=True,
    mlp_kwargs={
        "criterion": "gini",
        "max_depth": 10,
        "min_samples_split": 2,
    },
)
```

Notice how we haven't yet provide what is the data. We have simply outlined how the Classifier is to perform the classification and all of the underlying necessary steps.

## Declaring a Classifier

With our Pipeline properly set, we can declare our Classifier and pass it `my_pipeline`,

```python
from lexos.classification import Classifier
my_classifier = Classifier(
    train_data=MYDATA,
    labels=myLabels,
    pipeline=my_pipeline(),
)

classifier.fit()
predictions = classifier.predict(new_dataframe)
```

#### Interpreting Outputs

### Currently supported Pipelines

| Pipeline | Description | Best For |
|-----------|-------------|----------|
| `DecisionTreePipeline` | Non-parametric supervised learning algorithm | Purpose is to be a Decision Tree |


#### Custom Pipelines

If the available Pipelines don't match the users needs, it is possible for them to write their own one. To do so, the custom pipeline must subclass `Pipeline`, define the feature extraction rules by overwriting the `execute_training()` and `discover_features()` methods, and return a dictionary of trained components and evaluation metrics

```python
from lexos.classification.trainer import Pipeline

class CustomPipeline(Pipeline):
    """A custom pipeline strategy executing blank classification."""
    random_state: int = 42

    def discover_features(self, train_data: Any) -> list[str]:
        """Blank."""
        return list(train_data.columns)

    def execute_training(self, train_data: Any, labels: [Sequence[str]] = None,) -> dict[str, Any]:
        """Blank."""
        features = active_features or self.discover_features(train_data)
        X = train_data[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = A_NEW_MODEL_HERE()
        model.fit(X_scaled, labels)

        return {
            "final_model": model,
            "final_scaler": scaler,
            "holdout_metrics": {"accuracy": model.score(X_scaled, labels)},
            "holdout_report": pd.DataFrame(),
        }

        def predict(
        self, data: Any, results_payload: dict[str, Any], active_features: Optional[Sequence[str]] = None,) -> Sequence[str]:
        """Blank."""
        model = results_payload["final_model"]
        scaler = results_payload.get("final_scaler")

        features = active_features or list(data.columns)
        X = data[features]

        X_transformed = scaler.transform(X) if scaler is not None else X
        return model.predict(X_transformed)

```

#### Feature importance sweep
