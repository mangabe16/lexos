"""
Storage Abstraction Layer Prototype for Lexos Corpus Module

This prototype demonstrates how database storage can be integrated as a
supplementary enhancement to the existing file-based storage system.

Key Design Principles:
1. Zero impact on existing functionality
2. Optional database dependencies
3. Graceful fallback to file storage
4. Performance preservation
5. Flexible backend selection
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Iterator
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime
import json

# Import existing classes (would be actual imports in implementation)
# from lexos.corpus.record import Record
# from lexos.corpus.corpus import Corpus

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result of a storage query operation."""
    records: List['Record']
    total_count: int
    execution_time_ms: float
    backend_used: str

@dataclass 
class StorageMetrics:
    """Performance metrics for storage operations."""
    operation: str
    duration_ms: float
    records_processed: int
    memory_usage_mb: float
    backend: str

class StorageBackend(ABC):
    """Abstract base class for all storage backends."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the storage backend with configuration."""
        pass
    
    @abstractmethod
    def save_record(self, record: 'Record') -> bool:
        """Save a single record to storage."""
        pass
    
    @abstractmethod
    def load_record(self, record_id: str) -> Optional['Record']:
        """Load a single record by ID."""
        pass
    
    @abstractmethod
    def delete_record(self, record_id: str) -> bool:
        """Delete a record by ID."""
        pass
    
    @abstractmethod
    def query_records(self, query: Dict[str, Any]) -> QueryResult:
        """Query records by metadata criteria."""
        pass
    
    @abstractmethod
    def list_records(self, limit: Optional[int] = None, 
                    offset: Optional[int] = None) -> QueryResult:
        """List all records with optional pagination."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage backend statistics."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if backend is healthy and accessible."""
        pass
    
    @abstractmethod
    def backup(self, destination: Path) -> bool:
        """Create a backup of the storage."""
        pass
    
    @abstractmethod
    def restore(self, source: Path) -> bool:
        """Restore storage from backup."""
        pass

class FileStorageBackend(StorageBackend):
    """File-based storage backend (current implementation enhanced)."""
    
    def __init__(self):
        self.corpus_dir: Optional[Path] = None
        self.metadata_cache: Dict[str, Dict] = {}
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize file storage backend."""
        try:
            self.corpus_dir = Path(config.get('corpus_dir', 'corpus'))
            self.corpus_dir.mkdir(parents=True, exist_ok=True)
            (self.corpus_dir / 'data').mkdir(exist_ok=True)
            
            # Load metadata cache for faster queries
            self._load_metadata_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize file storage: {e}")
            return False
    
    def save_record(self, record: 'Record') -> bool:
        """Save record using existing to_disk() method."""
        try:
            # Use existing Record.to_disk() implementation
            filepath = self.corpus_dir / 'data' / f'{record.id}.bin'
            record.to_disk(str(filepath))
            
            # Update metadata cache
            self.metadata_cache[str(record.id)] = {
                'id': str(record.id),
                'name': record.name,
                'model': record.model,
                'is_active': record.is_active,
                'created_at': datetime.now().isoformat(),
                'filepath': str(filepath)
            }
            
            # Save updated metadata
            self._save_metadata_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to save record {record.id}: {e}")
            return False
    
    def load_record(self, record_id: str) -> Optional['Record']:
        """Load record using existing from_disk() method."""
        try:
            if record_id not in self.metadata_cache:
                return None
                
            filepath = self.metadata_cache[record_id]['filepath']
            # Use existing Record.from_disk() implementation
            return Record.from_disk(filepath)
        except Exception as e:
            logger.error(f"Failed to load record {record_id}: {e}")
            return None
    
    def delete_record(self, record_id: str) -> bool:
        """Delete record file and metadata."""
        try:
            if record_id in self.metadata_cache:
                filepath = Path(self.metadata_cache[record_id]['filepath'])
                if filepath.exists():
                    filepath.unlink()
                del self.metadata_cache[record_id]
                self._save_metadata_cache()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete record {record_id}: {e}")
            return False
    
    def query_records(self, query: Dict[str, Any]) -> QueryResult:
        """Query records using metadata cache."""
        start_time = datetime.now()
        
        try:
            matching_records = []
            
            for record_id, metadata in self.metadata_cache.items():
                if self._matches_query(metadata, query):
                    record = self.load_record(record_id)
                    if record:
                        matching_records.append(record)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return QueryResult(
                records=matching_records,
                total_count=len(matching_records),
                execution_time_ms=execution_time,
                backend_used='file'
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return QueryResult([], 0, 0, 'file')
    
    def list_records(self, limit: Optional[int] = None, 
                    offset: Optional[int] = None) -> QueryResult:
        """List all records with pagination."""
        record_ids = list(self.metadata_cache.keys())
        
        if offset:
            record_ids = record_ids[offset:]
        if limit:
            record_ids = record_ids[:limit]
        
        records = []
        for record_id in record_ids:
            record = self.load_record(record_id)
            if record:
                records.append(record)
        
        return QueryResult(
            records=records,
            total_count=len(self.metadata_cache),
            execution_time_ms=0,  # Would measure in real implementation
            backend_used='file'
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file storage statistics."""
        if not self.corpus_dir:
            return {}
            
        total_files = len(list((self.corpus_dir / 'data').glob('*.bin')))
        total_size = sum(f.stat().st_size for f in (self.corpus_dir / 'data').glob('*.bin'))
        
        return {
            'backend': 'file',
            'total_records': total_files,
            'total_size_bytes': total_size,
            'corpus_directory': str(self.corpus_dir),
            'metadata_cache_size': len(self.metadata_cache)
        }
    
    def health_check(self) -> bool:
        """Check file storage health."""
        return (self.corpus_dir and 
                self.corpus_dir.exists() and 
                (self.corpus_dir / 'data').exists())
    
    def backup(self, destination: Path) -> bool:
        """Create ZIP backup using existing save() method."""
        try:
            # Use existing Corpus.save() implementation
            # Would delegate to corpus.save(destination)
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def restore(self, source: Path) -> bool:
        """Restore from ZIP using existing load() method."""
        try:
            # Use existing Corpus.load() implementation  
            # Would delegate to corpus.load(source)
            self._load_metadata_cache()
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def _load_metadata_cache(self) -> None:
        """Load metadata cache from file."""
        cache_file = self.corpus_dir / 'metadata_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.metadata_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata cache: {e}")
                self.metadata_cache = {}
    
    def _save_metadata_cache(self) -> None:
        """Save metadata cache to file."""
        cache_file = self.corpus_dir / 'metadata_cache.json'
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.metadata_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata cache: {e}")
    
    def _matches_query(self, metadata: Dict, query: Dict[str, Any]) -> bool:
        """Check if metadata matches query criteria."""
        for key, value in query.items():
            if key not in metadata:
                return False
            if isinstance(value, dict):
                # Handle range queries: {'num_tokens': {'$gte': 100, '$lte': 1000}}
                if '$gte' in value and metadata[key] < value['$gte']:
                    return False
                if '$lte' in value and metadata[key] > value['$lte']:
                    return False
                if '$eq' in value and metadata[key] != value['$eq']:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True

class SQLiteStorageBackend(StorageBackend):
    """SQLite-enhanced storage backend."""
    
    def __init__(self):
        self.db_path: Optional[Path] = None
        self.file_backend: Optional[FileStorageBackend] = None
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize SQLite backend with file backend fallback."""
        try:
            import sqlite3
            
            self.db_path = Path(config.get('db_path', 'corpus.db'))
            
            # Initialize file backend for actual record storage
            self.file_backend = FileStorageBackend()
            if not self.file_backend.initialize(config):
                return False
            
            # Create SQLite database and tables
            self._create_database()
            return True
        except ImportError:
            logger.error("SQLite not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize SQLite storage: {e}")
            return False
    
    def save_record(self, record: 'Record') -> bool:
        """Save record to both file storage and SQLite index."""
        # Save to file storage first
        if not self.file_backend.save_record(record):
            return False
        
        # Index in SQLite
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO records 
                    (id, name, model, is_active, created_at, num_tokens, num_terms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(record.id), record.name, record.model, 
                    record.is_active, datetime.now().isoformat(),
                    record.num_tokens() if record.is_parsed else 0,
                    record.num_terms() if record.is_parsed else 0
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to index record in SQLite: {e}")
            return False
    
    def load_record(self, record_id: str) -> Optional['Record']:
        """Load record from file storage."""
        return self.file_backend.load_record(record_id)
    
    def delete_record(self, record_id: str) -> bool:
        """Delete from both file storage and SQLite."""
        if not self.file_backend.delete_record(record_id):
            return False
        
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete from SQLite: {e}")
            return False
    
    def query_records(self, query: Dict[str, Any]) -> QueryResult:
        """Fast querying using SQLite indexes."""
        start_time = datetime.now()
        
        try:
            import sqlite3
            
            # Build SQL query from dictionary
            sql, params = self._build_sql_query(query)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
            
            # Load actual records from file storage
            records = []
            for row in rows:
                record = self.load_record(row['id'])
                if record:
                    records.append(record)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return QueryResult(
                records=records,
                total_count=len(records),
                execution_time_ms=execution_time,
                backend_used='sqlite'
            )
        except Exception as e:
            logger.error(f"SQLite query failed, falling back to file storage: {e}")
            return self.file_backend.query_records(query)
    
    def list_records(self, limit: Optional[int] = None, 
                    offset: Optional[int] = None) -> QueryResult:
        """List records with SQLite pagination."""
        try:
            import sqlite3
            
            sql = "SELECT id FROM records ORDER BY created_at"
            params = []
            
            if limit:
                sql += " LIMIT ?"
                params.append(limit)
            if offset:
                sql += " OFFSET ?"
                params.append(offset)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(sql, params)
                record_ids = [row[0] for row in cursor.fetchall()]
                
                # Get total count
                total_count = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            
            records = []
            for record_id in record_ids:
                record = self.load_record(record_id)
                if record:
                    records.append(record)
            
            return QueryResult(
                records=records,
                total_count=total_count,
                execution_time_ms=0,  # Would measure in real implementation
                backend_used='sqlite'
            )
        except Exception as e:
            logger.error(f"SQLite list failed, falling back to file storage: {e}")
            return self.file_backend.list_records(limit, offset)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get SQLite storage statistics."""
        stats = self.file_backend.get_statistics()
        
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                record_count = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                stats.update({
                    'sqlite_records': record_count,
                    'sqlite_db_size_bytes': db_size,
                    'sqlite_db_path': str(self.db_path)
                })
        except Exception as e:
            logger.error(f"Failed to get SQLite statistics: {e}")
        
        return stats
    
    def health_check(self) -> bool:
        """Check SQLite and file storage health."""
        if not self.file_backend.health_check():
            return False
        
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def backup(self, destination: Path) -> bool:
        """Backup both file storage and SQLite database."""
        if not self.file_backend.backup(destination):
            return False
        
        # Copy SQLite database
        try:
            import shutil
            db_backup = destination.parent / f"{destination.stem}_sqlite.db"
            shutil.copy2(self.db_path, db_backup)
            return True
        except Exception as e:
            logger.error(f"SQLite backup failed: {e}")
            return False
    
    def restore(self, source: Path) -> bool:
        """Restore both file storage and SQLite database."""
        if not self.file_backend.restore(source):
            return False
        
        # Restore SQLite database
        try:
            import shutil
            db_backup = source.parent / f"{source.stem}_sqlite.db" 
            if db_backup.exists():
                shutil.copy2(db_backup, self.db_path)
            else:
                # Rebuild SQLite index from restored files
                self._rebuild_index()
            return True
        except Exception as e:
            logger.error(f"SQLite restore failed: {e}")
            return False
    
    def _create_database(self) -> None:
        """Create SQLite tables and indexes."""
        import sqlite3
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model TEXT,
                    is_active BOOLEAN,
                    created_at TEXT,
                    num_tokens INTEGER,
                    num_terms INTEGER,
                    vocabulary_density REAL
                )
            """)
            
            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_records_name ON records(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_records_model ON records(model)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_records_active ON records(is_active)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_records_created ON records(created_at)")
            
            conn.commit()
    
    def _build_sql_query(self, query: Dict[str, Any]) -> tuple[str, list]:
        """Convert dictionary query to SQL."""
        conditions = []
        params = []
        
        for key, value in query.items():
            if isinstance(value, dict):
                # Handle range queries
                if '$gte' in value:
                    conditions.append(f"{key} >= ?")
                    params.append(value['$gte'])
                if '$lte' in value:
                    conditions.append(f"{key} <= ?")
                    params.append(value['$lte'])
                if '$eq' in value:
                    conditions.append(f"{key} = ?")
                    params.append(value['$eq'])
            else:
                conditions.append(f"{key} = ?")
                params.append(value)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT id FROM records WHERE {where_clause}"
        
        return sql, params
    
    def _rebuild_index(self) -> None:
        """Rebuild SQLite index from file storage."""
        # Would iterate through file storage and re-index all records
        pass

class HybridStorageManager:
    """Manages multiple storage backends with intelligent fallback."""
    
    def __init__(self, primary: StorageBackend, 
                 secondary: Optional[StorageBackend] = None):
        self.primary = primary
        self.secondary = secondary
        self.metrics: List[StorageMetrics] = []
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize all configured backends."""
        success = self.primary.initialize(config)
        
        if self.secondary:
            try:
                secondary_success = self.secondary.initialize(config)
                if not secondary_success:
                    logger.warning("Secondary storage backend failed to initialize")
                    self.secondary = None
            except Exception as e:
                logger.warning(f"Secondary storage initialization failed: {e}")
                self.secondary = None
        
        return success
    
    def save_record(self, record: 'Record') -> bool:
        """Save to primary, sync to secondary if available."""
        start_time = datetime.now()
        
        # Always save to primary
        success = self.primary.save_record(record)
        
        # Sync to secondary if available
        if success and self.secondary:
            try:
                self.secondary.save_record(record)
            except Exception as e:
                logger.warning(f"Secondary storage sync failed: {e}")
        
        self._record_metric('save', start_time, 1, 
                          'primary' + ('+secondary' if self.secondary else ''))
        return success
    
    def load_record(self, record_id: str) -> Optional['Record']:
        """Load from fastest available backend."""
        start_time = datetime.now()
        
        # Try secondary first (usually faster for queries)
        if self.secondary and self.secondary.health_check():
            try:
                record = self.secondary.load_record(record_id)
                if record:
                    self._record_metric('load', start_time, 1, 'secondary')
                    return record
            except Exception as e:
                logger.warning(f"Secondary storage load failed: {e}")
        
        # Fallback to primary
        record = self.primary.load_record(record_id)
        self._record_metric('load', start_time, 1, 'primary')
        return record
    
    def query_records(self, query: Dict[str, Any]) -> QueryResult:
        """Use best backend for querying."""
        # Prefer secondary for queries (usually has indexing)
        if self.secondary and self.secondary.health_check():
            try:
                return self.secondary.query_records(query)
            except Exception as e:
                logger.warning(f"Secondary storage query failed: {e}")
        
        # Fallback to primary
        return self.primary.query_records(query)
    
    def get_backend_status(self) -> Dict[str, Any]:
        """Get status of all backends."""
        status = {
            'primary': {
                'type': type(self.primary).__name__,
                'healthy': self.primary.health_check(),
                'statistics': self.primary.get_statistics()
            }
        }
        
        if self.secondary:
            status['secondary'] = {
                'type': type(self.secondary).__name__,
                'healthy': self.secondary.health_check(),
                'statistics': self.secondary.get_statistics()
            }
        
        return status
    
    def get_performance_metrics(self) -> List[StorageMetrics]:
        """Get recent performance metrics."""
        return self.metrics[-100:]  # Return last 100 operations
    
    def _record_metric(self, operation: str, start_time: datetime, 
                      records_processed: int, backend: str) -> None:
        """Record performance metric."""
        duration = (datetime.now() - start_time).total_seconds() * 1000
        
        metric = StorageMetrics(
            operation=operation,
            duration_ms=duration,
            records_processed=records_processed,
            memory_usage_mb=0,  # Would use memory_profiler in real implementation
            backend=backend
        )
        
        self.metrics.append(metric)
        
        # Keep only recent metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-100:]

# Factory function for creating storage managers
def create_storage_manager(storage_type: str = 'file', 
                          config: Optional[Dict[str, Any]] = None) -> HybridStorageManager:
    """Factory function to create appropriate storage manager."""
    if config is None:
        config = {}
    
    if storage_type == 'file':
        primary = FileStorageBackend()
        return HybridStorageManager(primary)
    
    elif storage_type == 'sqlite':
        primary = FileStorageBackend()
        secondary = SQLiteStorageBackend()
        return HybridStorageManager(primary, secondary)
    
    elif storage_type == 'mongodb':
        # Would implement MongoStorageBackend
        primary = FileStorageBackend()
        # secondary = MongoStorageBackend()
        return HybridStorageManager(primary)
    
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")

# Example usage
if __name__ == "__main__":
    # Create SQLite-enhanced storage
    storage_manager = create_storage_manager('sqlite', {
        'corpus_dir': 'test_corpus',
        'db_path': 'test_corpus.db'
    })
    
    # Initialize
    if storage_manager.initialize({'corpus_dir': 'test_corpus'}):
        print("Storage manager initialized successfully")
        
        # Check status
        status = storage_manager.get_backend_status()
        print(f"Backend status: {status}")
    else:
        print("Failed to initialize storage manager")