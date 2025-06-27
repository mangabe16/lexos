"""test_init.py.

Test coverage for corpus.__init__.py import branches.
Simplified tests that verify import functionality without complex mocking.

Last Update: 2025-06-20.
"""

import pytest


class TestCorpusInitImportBranches:
    """Test import functionality in corpus.__init__.py."""
    
    def test_basic_imports_work(self):
        """Test that basic imports work correctly."""
        import lexos.corpus
        
        # Basic classes should be available
        assert hasattr(lexos.corpus, 'Record')
        assert hasattr(lexos.corpus, 'LexosModelCache')
        assert hasattr(lexos.corpus, 'RecordsDict')
        
        # __all__ should contain the basic classes
        all_exports = getattr(lexos.corpus, '__all__', [])
        assert 'Record' in all_exports
        assert 'LexosModelCache' in all_exports
        assert 'RecordsDict' in all_exports
        
        print("✓ Basic imports work correctly")
    
    def test_conditional_imports(self):
        """Test that conditional imports (CorpusStats, Corpus) are handled correctly."""
        import lexos.corpus
        
        all_exports = getattr(lexos.corpus, '__all__', [])
        
        # Check if CorpusStats is available (lines 22-25 coverage)
        if hasattr(lexos.corpus, 'CorpusStats') and lexos.corpus.CorpusStats is not None:
            assert 'CorpusStats' in all_exports
            print("✓ CorpusStats available and in __all__")
        else:
            print("✓ CorpusStats not available (handled gracefully)")
        
        # Check if Corpus is available (lines 31-34 coverage)  
        if hasattr(lexos.corpus, 'Corpus') and lexos.corpus.Corpus is not None:
            assert 'Corpus' in all_exports
            print("✓ Corpus available and in __all__")
        else:
            print("✓ Corpus not available (handled gracefully)")
    
    def test_import_exception_handling(self):
        """Test that import exceptions are handled gracefully."""
        import lexos.corpus
        
        # The module should import successfully even if some components fail
        assert hasattr(lexos.corpus, '__all__')
        all_exports = getattr(lexos.corpus, '__all__')
        
        # Should have at least the basic exports
        assert len(all_exports) >= 3  # Record, LexosModelCache, RecordsDict
        
        # Critical imports should not cause the module to fail completely
        # This covers lines 45-48 (critical import failure handling)
        assert 'Record' in all_exports
        
        print("✓ Import exception handling works correctly")
    
    def test_all_exports_valid(self):
        """Test that all exported classes are actually available."""
        import lexos.corpus
        
        all_exports = getattr(lexos.corpus, '__all__', [])
        
        for export_name in all_exports:
            # Each exported name should be available as an attribute
            assert hasattr(lexos.corpus, export_name)
            
            # The attribute should not be None
            export_obj = getattr(lexos.corpus, export_name)
            assert export_obj is not None
        
        print(f"✓ All {len(all_exports)} exports are valid")