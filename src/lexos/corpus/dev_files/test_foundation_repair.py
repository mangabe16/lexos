#!/usr/bin/env python3
"""
Test script for foundation repair and communication architecture.
"""

import tempfile
import traceback

def test_corpus_stats():
    """Test CorpusStats instantiation and new functionality."""
    print("🔍 Testing CorpusStats...")
    try:
        from lexos.corpus.corpus_stats import CorpusStats
        
        # Create more realistic test data for better statistical analysis
        sample_docs = [
            ('doc1', 'Document 1', ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']),
            ('doc2', 'Document 2', ['hello', 'world', 'this', 'is', 'a', 'test', 'document', 'with', 'some', 'words']),
            ('doc3', 'Document 3', ['another', 'example', 'text', 'for', 'analysis', 'purposes', 'with', 'different', 'vocabulary']),
            ('doc4', 'Document 4', ['corpus', 'linguistics', 'text', 'analysis', 'statistical', 'methods', 'research', 'academic', 'study']),
            ('doc5', 'Document 5', ['natural', 'language', 'processing', 'computational', 'linguistics', 'machine', 'learning', 'algorithms', 'data', 'science'])
        ]
        
        stats = CorpusStats(docs=sample_docs)
        print('✅ CorpusStats instantiation SUCCESSFUL')
        
        # Test basic stats
        print(f'📊 Mean: {stats.mean:.2f}')
        print(f'📊 Std Dev: {stats.standard_deviation:.2f}')
        
        # Test new distribution stats
        dist_stats = stats.distribution_stats
        print(f'📈 Skewness: {dist_stats["skewness"]:.3f}')
        print(f'📈 Kurtosis: {dist_stats["kurtosis"]:.3f}')
        print(f'📈 Is Normal: {dist_stats["is_normal"]}')
        
        # Test new text diversity stats
        diversity = stats.text_diversity_stats
        print(f'📝 Mean TTR: {diversity["mean_ttr"]:.3f}')
        print(f'📝 Corpus TTR: {diversity["corpus_ttr"]:.3f}')
        
        # Test group comparison
        comparison = stats.compare_groups(['Document 1'], ['Document 2', 'Document 3'])
        print(f'🔬 Group comparison p-value: {comparison["p_value"]:.3f}')
        
        # Test bootstrap CI
        bootstrap = stats.bootstrap_confidence_interval(n_bootstrap=100)
        print(f'🔄 Bootstrap CI: [{bootstrap["ci_lower"]:.2f}, {bootstrap["ci_upper"]:.2f}]')
        
        # ===== NEW PHASE 2 FEATURES =====
        print('\n🎯 Testing Phase 2: Text-Specific Statistics...')
        
        # Test advanced lexical diversity
        adv_diversity = stats.advanced_lexical_diversity
        print(f'📊 Advanced Diversity - Mean CTTR: {adv_diversity["mean_cttr"]:.3f}')
        print(f'📊 Advanced Diversity - Mean RTTR: {adv_diversity["mean_rttr"]:.3f}')
        print(f'📊 Advanced Diversity - Range: {adv_diversity["diversity_range"]:.3f}')
        
        # Test Zipf analysis
        zipf = stats.zipf_analysis
        print(f'📈 Zipf Analysis - Slope: {zipf["zipf_slope"]:.3f}')
        print(f'📈 Zipf Analysis - R²: {zipf["r_squared"]:.3f}')
        print(f'📈 Zipf Analysis - Follows Zipf: {zipf["follows_zipf"]}')
        print(f'📈 Zipf Analysis - Quality: {zipf["zipf_goodness_of_fit"]}')
        
        # Test corpus quality metrics
        quality = stats.corpus_quality_metrics
        print(f'🏆 Corpus Quality - Length Balance: {quality["document_length_balance"]["classification"]}')
        print(f'🏆 Corpus Quality - Sampling Adequacy: {quality["vocabulary_richness"]["sampling_adequacy"]}')
        print(f'🏆 Corpus Quality - Size Adequacy: {quality["corpus_size_metrics"]["size_adequacy"]}')
        
        # Test enhanced text diversity with dislegomena
        diversity = stats.text_diversity_stats
        print(f'📝 Enhanced Diversity - Hapax Ratio: {diversity["corpus_hapax_ratio"]:.3f}')
        print(f'📝 Enhanced Diversity - Dislegomena Ratio: {diversity["corpus_dislegomena_ratio"]:.3f}')
        
        print('✅ All Phase 2 text-specific features working!')
        print('✅ All CorpusStats features working!\n')
        return True
        
    except Exception as e:
        print(f'❌ CorpusStats FAILED: {e}')
        traceback.print_exc()
        return False

def test_communication_architecture():
    """Test communication architecture framework."""
    print("🔗 Testing Communication Architecture...")
    try:
        from lexos.corpus.corpus import Corpus
        
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Corpus(name='Test Corpus', corpus_dir=temp_dir)
            corpus.add('Hello world document', name='doc1')
            corpus.add('Another test document', name='doc2')
            
            # Test statistical fingerprint export
            fingerprint = corpus.export_statistical_fingerprint()
            print('✅ Statistical fingerprint export SUCCESSFUL')
            print(f'🔍 Corpus fingerprint: {fingerprint["corpus_metadata"]["corpus_fingerprint"]}')
            print(f'📊 Num docs: {fingerprint["corpus_metadata"]["num_docs"]}')
            
            # Test analysis result import
            test_results = {
                'clusters': [0, 1],
                'silhouette_score': 0.75,
                'analysis_type': 'kmeans'
            }
            corpus.import_analysis_results('kmeans', test_results, version='1.0.0')
            print('✅ Analysis result import SUCCESSFUL')
            
            # Test result retrieval
            stored_results = corpus.get_analysis_results('kmeans')
            print(f'📥 Stored results version: {stored_results["version"]}')
            print(f'📥 Results data: {stored_results["results"]["analysis_type"]}')
            
            # Test compatibility validation
            compatibility = corpus.validate_analysis_compatibility('kmeans')
            print(f'🔒 Analysis compatibility: {compatibility["compatible"]}')
            
            print('✅ All communication features working!\n')
            return True
            
    except Exception as e:
        print(f'❌ Communication architecture FAILED: {e}')
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Foundation Repair Implementation\n")
    
    corpus_stats_success = test_corpus_stats()
    comm_arch_success = test_communication_architecture()
    
    print("📋 SUMMARY:")
    print(f"   CorpusStats: {'✅ PASS' if corpus_stats_success else '❌ FAIL'}")
    print(f"   Communication: {'✅ PASS' if comm_arch_success else '❌ FAIL'}")
    
    if corpus_stats_success and comm_arch_success:
        print("\n🎉 Foundation Repair COMPLETE - Ready for Phase 2!")
    else:
        print("\n🔧 Some issues need fixing before proceeding.")

if __name__ == "__main__":
    main()