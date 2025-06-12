"""__init__.py.

Corpus module initialization - exposes main classes for import.
This fixes the import issue in corpus.py line 33.

Last updated: June 12, 2025
"""

# Import main classes to make them available at package level
# Import order matters to avoid circular imports
try:
    # Import Record first (no dependencies on other corpus modules)
    from lexos.corpus.record import Record
    
    # Import utils (no dependencies on other corpus modules)
    from lexos.corpus.utils import LexosModelCache, RecordsDict
    
    # Try to import CorpusStats (depends on DTM which might have issues)
    try:
        from lexos.corpus.corpus_stats import CorpusStats
        corpus_stats_available = True
    except Exception as e:
        print(f"Warning: CorpusStats not available: {e}")
        CorpusStats = None
        corpus_stats_available = False
    
    # Import Corpus last (depends on Record import working)
    try:
        from lexos.corpus.corpus import Corpus
        corpus_available = True
    except Exception as e:
        print(f"Warning: Corpus class not available: {e}")
        Corpus = None
        corpus_available = False
    
    # Define what's available for import
    __all__ = ["Record", "LexosModelCache", "RecordsDict"]
    
    if corpus_stats_available:
        __all__.append("CorpusStats")
    
    if corpus_available:
        __all__.append("Corpus")
    
except ImportError as e:
    # If basic imports fail, define minimal __all__
    __all__ = []
    print(f"Critical error: Failed to import basic corpus classes: {e}")

