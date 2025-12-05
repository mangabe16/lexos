
# Mallet

Topic modeling is a statistical method for discovering abstract themes or "topics" within a collection of documents. MALLET is a mature tool for topic modeling used widely in the Humanities. It is a Java package that needs to be installed separately from Lexos. The Lexos `mallet` module provides a straightforward wrapper for running MALLET, managing outputs, and creating visualizations of your topic model.

::: lexos.topic_modeling.mallet
  handler: python
  selection:
    members:
      - MALLET_BINARY_PATH
      - Mallet
      - Mallet.read_file
      - Mallet.read_dirs
      - Mallet.import_files
      - Mallet.import_docs
      - Mallet._check_format
      - Mallet._metadata_get
      - Mallet._metadata_has
      - Mallet._import_training_data
      - Mallet._setup_wordcloud
      - Mallet._track_progress
