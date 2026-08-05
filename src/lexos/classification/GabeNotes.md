# To-do
- BIG refactor
    - Currently, train_data is EITHER text or a pd.DataFrame, meaning discoer_features() can EITHER be bag of words or the CorpusStats extarcted features

- Need to finish refactoring trainer.py
    - Need to allow for the feature removal sweep to have different removal algorithms, as well as permutations of different features at a time, and rmeoving more than one at a time. Also consider an additive approach
        - Need to read acquired literature on feature removal algorithms
        - This is currently an enhancement issue on github
    - Do I need features attribute in Classifier? I believe not, must investigate


- Need to finish documentation
    - Need to write an user guide and finish these notes
        - Should Classification be added under Analysis?
    - Need to write a proper tutorial for classification and for mlp
    - Need to write API documentation


- Investigate if I should be using StandardScaler so much, and if we should offer different ones

- NEED TO CHECK RANDOM LIBRARY AND HOW THE SEED IS BEING SET
    Using Fisher-Yates Shuffle algorithm

- Need to check how model metrics are being output
    Currently the f1-score impacted is subtracted from the baseline, but should it be compared to the prior model performance?

- It might also be possible to add a method for saving and loading the model, but this will depend on the specific model being trained and how it handles serialization. It may be better to handle this separately for each model type, rather than trying to include it in the base Classifier class.
- Trained models could be passed to visualisation functions tuned to the particular type of model.
