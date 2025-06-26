"""
Lexos Package

This file makes core modules and classes available at the package level,
so you can do imports like:

    from lexos import KMeansCluster
"""

from .kmeans import KMeansCluster

# If you have other core components, you can expose them here too:
# from .dtm import DTM
# from .exceptions import LexosException

__all__ = [
    "KMeansCluster",
    # "DTM",
    # "LexosException",
]
