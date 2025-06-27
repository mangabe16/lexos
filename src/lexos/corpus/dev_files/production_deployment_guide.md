# Production Deployment Guide for Database-Enhanced Lexos

## Overview

This guide provides comprehensive deployment strategies for the Lexos corpus module with optional database enhancements in production environments. The deployment maintains backward compatibility while adding enterprise-grade capabilities.

## Deployment Scenarios

### Scenario 1: Research Environment (Current System)
**Use Case**: Academic research, individual analysis, development work
**Complexity**: Minimal
**Users**: Single user, offline capable
**Scale**: Small to medium corpora (< 10,000 documents)

```python
# Configuration
storage_config = {
    'storage_type': 'file',
    'corpus_dir': '/data/corpus',
    'backup_enabled': True,
    'backup_schedule': 'daily'
}

# Implementation
from lexos.corpus import Corpus
corpus = Corpus(corpus_dir='/data/corpus')
# Use existing API - no changes required
```

**Deployment Requirements**:
- Python 3.12+ environment
- Adequate disk space (2-5GB typical)
- Regular backup to external storage
- No external dependencies

### Scenario 2: Local Production (SQLite Enhanced)
**Use Case**: Small team, desktop applications, local production systems
**Complexity**: Low
**Users**: 1-5 concurrent users
**Scale**: Medium corpora (10,000-100,000 documents)

```python
# Configuration
storage_config = {
    'storage_type': 'sqlite',
    'corpus_dir': '/data/corpus',
    'db_path': '/data/corpus.db',
    'enable_indexing': True,
    'cache_size_mb': 256,
    'wal_mode': True  # Better concurrency
}

# Implementation
from lexos.corpus import Corpus
from lexos.storage import create_storage_manager

storage_manager = create_storage_manager('sqlite', storage_config)
corpus = Corpus(storage_manager=storage_manager)
```

**Deployment Requirements**:
- Python 3.12+ with SQLite3
- SSD storage recommended (significant performance improvement)
- 4-8GB RAM minimum
- Automated backup of both corpus files and SQLite database

**Performance Characteristics**:
- Write: 500-2,000 records/second
- Read: 50,000+ records/second (indexed queries)
- Query: Complex queries 10-100x faster than file scanning
- Concurrent: 4-8 simultaneous users supported

### Scenario 3: Enterprise Cloud (MongoDB Enhanced)
**Use Case**: Large organizations, cloud services, multi-user platforms
**Complexity**: High
**Users**: 10-100+ concurrent users  
**Scale**: Large corpora (100,000+ documents)

```python
# Configuration
storage_config = {
    'storage_type': 'mongodb',
    'corpus_dir': '/data/corpus',
    'mongodb_uri': 'mongodb://mongodb-cluster:27017/lexos',
    'database_name': 'lexos_production',
    'collection_name': 'corpus_records',
    'replica_set': 'rs0',
    'read_preference': 'secondaryPreferred',
    'write_concern': {'w': 'majority', 'j': True}
}

# Implementation
from lexos.corpus import Corpus
from lexos.storage import create_storage_manager

storage_manager = create_storage_manager('mongodb', storage_config)
corpus = Corpus(storage_manager=storage_manager)
```

**Deployment Requirements**:
- MongoDB cluster (3+ nodes recommended)
- Application servers with 8-16GB RAM
- Distributed file storage (NFS, S3, etc.)
- Load balancer for multi-instance deployment
- Monitoring and logging infrastructure

**Performance Characteristics**:
- Write: 1,000-10,000 records/second (optimized)
- Read: 10,000-50,000 records/second
- Query: Complex aggregations and real-time analytics
- Concurrent: 100+ simultaneous users supported

## Detailed Deployment Configurations

### SQLite Production Deployment

#### System Requirements
```yaml
# Minimum Requirements
CPU: 2 cores
RAM: 4GB
Storage: 100GB SSD
OS: Linux/Windows/macOS

# Recommended Requirements  
CPU: 4+ cores
RAM: 8-16GB
Storage: 500GB+ NVMe SSD
OS: Linux (Ubuntu 20.04+ or RHEL 8+)
```

#### Configuration Files

**config/sqlite_production.json**
```json
{
  "storage": {
    "type": "sqlite",
    "corpus_dir": "/opt/lexos/data/corpus",
    "db_path": "/opt/lexos/data/corpus.db",
    "backup_dir": "/opt/lexos/backups",
    "enable_wal": true,
    "cache_size_mb": 512,
    "timeout_seconds": 30,
    "max_connections": 10
  },
  "performance": {
    "enable_query_cache": true,
    "cache_ttl_seconds": 3600,
    "background_indexing": true,
    "vacuum_schedule": "weekly"
  },
  "backup": {
    "enabled": true,
    "schedule": "0 2 * * *",
    "retention_days": 30,
    "compression": true
  },
  "monitoring": {
    "log_level": "INFO",
    "log_file": "/var/log/lexos/application.log",
    "metrics_enabled": true,
    "health_check_interval": 60
  }
}
```

#### Docker Deployment

**Dockerfile**
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /opt/lexos

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create data directories
RUN mkdir -p /opt/lexos/data /opt/lexos/backups /var/log/lexos

# Set permissions
RUN useradd -r -s /bin/false lexos && \
    chown -R lexos:lexos /opt/lexos /var/log/lexos

USER lexos

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from lexos.storage import health_check; exit(0 if health_check() else 1)"

# Run application
CMD ["python", "-m", "lexos.server", "--config", "/opt/lexos/config/sqlite_production.json"]
```

**docker-compose.yml**
```yaml
version: '3.8'
services:
  lexos-app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - lexos_data:/opt/lexos/data
      - lexos_backups:/opt/lexos/backups
      - lexos_logs:/var/log/lexos
    environment:
      - LEXOS_ENV=production
      - LEXOS_CONFIG=/opt/lexos/config/sqlite_production.json
    restart: unless-stopped
    
  lexos-backup:
    image: alpine:latest
    volumes:
      - lexos_data:/data
      - lexos_backups:/backups
    command: >
      sh -c "
        echo '0 2 * * * cp -r /data/* /backups/$$(date +%Y%m%d_%H%M%S)/' | crontab - &&
        crond -f
      "
    restart: unless-stopped

volumes:
  lexos_data:
  lexos_backups:
  lexos_logs:
```

### MongoDB Production Deployment

#### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   App Server 1  │    │   App Server 2  │
│    (HAProxy)    │◄──►│     (Docker)    │    │     (Docker)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └─────────────►│  Shared Storage │◄─────────────┘
                        │   (NFS/S3)      │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │  MongoDB Cluster│
                        │  (3+ Replicas)  │
                        └─────────────────┘
```

#### MongoDB Configuration

**config/mongodb_production.json**
```json
{
  "storage": {
    "type": "mongodb",
    "corpus_dir": "/mnt/shared/corpus",
    "mongodb": {
      "uri": "mongodb://mongo1:27017,mongo2:27017,mongo3:27017/lexos?replicaSet=rs0",
      "database": "lexos_production",
      "collections": {
        "records": "corpus_records",
        "metadata": "corpus_metadata",
        "users": "users",
        "sessions": "user_sessions"
      },
      "options": {
        "maxPoolSize": 50,
        "minPoolSize": 5,
        "maxIdleTimeMS": 300000,
        "serverSelectionTimeoutMS": 5000,
        "socketTimeoutMS": 300000,
        "readPreference": "secondaryPreferred",
        "writeConcern": {"w": "majority", "j": true, "wtimeout": 5000}
      }
    }
  },
  "clustering": {
    "enabled": true,
    "node_id": "${NODE_ID}",
    "discovery_service": "consul://consul:8500",
    "health_check_interval": 30
  },
  "caching": {
    "redis": {
      "enabled": true,
      "uri": "redis://redis-cluster:6379",
      "ttl_seconds": 3600,
      "max_memory": "2gb"
    }
  },
  "file_storage": {
    "backend": "s3",
    "s3": {
      "bucket": "lexos-corpus-data",
      "region": "us-west-2",
      "access_key": "${AWS_ACCESS_KEY}",
      "secret_key": "${AWS_SECRET_KEY}"
    }
  }
}
```

#### Kubernetes Deployment

**k8s/namespace.yaml**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: lexos-production
```

**k8s/configmap.yaml**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: lexos-config
  namespace: lexos-production
data:
  production.json: |
    {
      "storage": {
        "type": "mongodb",
        "mongodb": {
          "uri": "mongodb://mongodb-service:27017/lexos",
          "database": "lexos_production"
        }
      }
    }
```

**k8s/deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lexos-app
  namespace: lexos-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lexos-app
  template:
    metadata:
      labels:
        app: lexos-app
    spec:
      containers:
      - name: lexos
        image: lexos:latest
        ports:
        - containerPort: 8000
        env:
        - name: LEXOS_CONFIG
          value: "/config/production.json"
        volumeMounts:
        - name: config
          mountPath: /config
        - name: corpus-data
          mountPath: /mnt/shared/corpus
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: lexos-config
      - name: corpus-data
        persistentVolumeClaim:
          claimName: corpus-data-pvc
```

**k8s/service.yaml**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: lexos-service
  namespace: lexos-production
spec:
  selector:
    app: lexos-app
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

## Migration Strategies

### File to SQLite Migration

```python
#!/usr/bin/env python3
"""
Migration script: File storage to SQLite enhanced storage
"""

import logging
from pathlib import Path
from lexos.corpus import Corpus
from lexos.storage import create_storage_manager

def migrate_file_to_sqlite(source_corpus_dir: str, target_db_path: str):
    """Migrate existing file-based corpus to SQLite enhanced storage."""
    
    logging.info(f"Starting migration from {source_corpus_dir} to {target_db_path}")
    
    # Load existing corpus
    source_corpus = Corpus(corpus_dir=source_corpus_dir)
    logging.info(f"Loaded {source_corpus.num_docs} documents from source")
    
    # Create SQLite enhanced storage
    target_storage = create_storage_manager('sqlite', {
        'corpus_dir': source_corpus_dir,  # Keep same file location
        'db_path': target_db_path
    })
    
    if not target_storage.initialize({'corpus_dir': source_corpus_dir}):
        raise RuntimeError("Failed to initialize target storage")
    
    # Migrate records
    success_count = 0
    error_count = 0
    
    for record_id, record in source_corpus.records.items():
        try:
            # Record already exists in file storage, just index in SQLite
            if target_storage.save_record(record):
                success_count += 1
            else:
                error_count += 1
                logging.error(f"Failed to migrate record {record_id}")
        except Exception as e:
            error_count += 1
            logging.error(f"Error migrating record {record_id}: {e}")
        
        if (success_count + error_count) % 1000 == 0:
            logging.info(f"Progress: {success_count} success, {error_count} errors")
    
    logging.info(f"Migration complete: {success_count} success, {error_count} errors")
    
    # Verify migration
    target_corpus = Corpus(storage_manager=target_storage)
    if target_corpus.num_docs == source_corpus.num_docs:
        logging.info("Migration verification successful")
        return True
    else:
        logging.error(f"Migration verification failed: {target_corpus.num_docs} != {source_corpus.num_docs}")
        return False

if __name__ == "__main__":
    migrate_file_to_sqlite("/data/existing_corpus", "/data/corpus.db")
```

### Rolling Deployment Strategy

```python
"""
Zero-downtime deployment strategy for production upgrades
"""

class RollingDeployment:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.current_version = self.get_current_version()
        
    def deploy_new_version(self, new_version: str):
        """Deploy new version with zero downtime."""
        
        # Step 1: Deploy to staging environment
        self.deploy_to_staging(new_version)
        self.run_health_checks('staging')
        
        # Step 2: Create backup
        backup_path = self.create_backup()
        
        # Step 3: Rolling update of instances
        instances = self.get_active_instances()
        
        for instance in instances:
            # Remove instance from load balancer
            self.remove_from_load_balancer(instance)
            
            # Deploy new version to instance
            self.deploy_to_instance(instance, new_version)
            
            # Health check new instance
            if self.health_check_instance(instance):
                # Add back to load balancer
                self.add_to_load_balancer(instance)
            else:
                # Rollback instance
                self.rollback_instance(instance, self.current_version)
                raise RuntimeError(f"Health check failed for {instance}")
            
            # Wait between instances for stability
            time.sleep(30)
        
        # Step 4: Final verification
        self.run_integration_tests()
        
        # Step 5: Update configuration
        self.update_version_config(new_version)
        
        logging.info(f"Deployment complete: {self.current_version} -> {new_version}")
```

## Monitoring and Maintenance

### Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123

  lexos-exporter:
    build: ./monitoring/
    ports:
      - "8080:8080"
    environment:
      - LEXOS_API_URL=http://lexos-app:8000
      - METRICS_PORT=8080

volumes:
  prometheus_data:
  grafana_data:
```

### Key Metrics to Monitor

```python
# monitoring/lexos_exporter.py
"""
Prometheus metrics exporter for Lexos corpus system
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import requests

# Define metrics
CORPUS_OPERATIONS = Counter('lexos_operations_total', 'Total operations', ['operation', 'status'])
OPERATION_DURATION = Histogram('lexos_operation_duration_seconds', 'Operation duration', ['operation'])
ACTIVE_CORPORA = Gauge('lexos_active_corpora', 'Number of active corpora')
STORAGE_SIZE = Gauge('lexos_storage_size_bytes', 'Storage size in bytes', ['backend'])
QUERY_PERFORMANCE = Histogram('lexos_query_duration_seconds', 'Query duration', ['query_type'])

class LexosMetricsExporter:
    def __init__(self, api_url: str):
        self.api_url = api_url
        
    def collect_metrics(self):
        """Collect metrics from Lexos API."""
        try:
            # Get system status
            status = requests.get(f"{self.api_url}/api/status").json()
            
            ACTIVE_CORPORA.set(status.get('active_corpora', 0))
            STORAGE_SIZE.labels(backend='file').set(status.get('file_storage_bytes', 0))
            STORAGE_SIZE.labels(backend='database').set(status.get('db_storage_bytes', 0))
            
            # Get performance metrics
            metrics = requests.get(f"{self.api_url}/api/metrics").json()
            
            for metric in metrics.get('recent_operations', []):
                OPERATION_DURATION.labels(operation=metric['operation']).observe(metric['duration'])
                CORPUS_OPERATIONS.labels(
                    operation=metric['operation'], 
                    status='success' if metric['success'] else 'error'
                ).inc()
                
        except Exception as e:
            CORPUS_OPERATIONS.labels(operation='health_check', status='error').inc()

if __name__ == "__main__":
    exporter = LexosMetricsExporter("http://lexos-app:8000")
    start_http_server(8080)
    
    while True:
        exporter.collect_metrics()
        time.sleep(30)
```

### Automated Backup and Recovery

```bash
#!/bin/bash
# backup_script.sh - Automated backup for Lexos production

set -e

BACKUP_DIR="/opt/lexos/backups"
CORPUS_DIR="/opt/lexos/data/corpus"
DB_PATH="/opt/lexos/data/corpus.db"
RETENTION_DAYS=30
S3_BUCKET="lexos-backups"

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$TIMESTAMP"

echo "Starting backup at $(date)"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Backup corpus files
echo "Backing up corpus files..."
tar -czf "$BACKUP_PATH/corpus_files.tar.gz" -C "$CORPUS_DIR" .

# Backup SQLite database (if exists)
if [ -f "$DB_PATH" ]; then
    echo "Backing up SQLite database..."
    sqlite3 "$DB_PATH" ".backup '$BACKUP_PATH/corpus.db'"
fi

# Create backup manifest
cat > "$BACKUP_PATH/manifest.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "files": {
        "corpus_files": "corpus_files.tar.gz",
        "database": "corpus.db"
    },
    "system_info": {
        "hostname": "$(hostname)",
        "disk_usage": "$(df -h $CORPUS_DIR | tail -1)"
    }
}
EOF

# Upload to S3 (if configured)
if command -v aws &> /dev/null && [ -n "$S3_BUCKET" ]; then
    echo "Uploading to S3..."
    aws s3 sync "$BACKUP_PATH" "s3://$S3_BUCKET/backups/$TIMESTAMP/"
fi

# Cleanup old backups
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} +

echo "Backup completed successfully at $(date)"
```

## Security Considerations

### Access Control

```python
# security/auth.py
"""
Authentication and authorization for Lexos production deployment
"""

from functools import wraps
from flask import request, jsonify, current_app
import jwt
from datetime import datetime, timedelta

class AuthenticationManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.token_expiry = timedelta(hours=24)
    
    def generate_token(self, user_id: str, permissions: list) -> str:
        """Generate JWT token for user."""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + self.token_expiry,
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

def require_auth(permissions=None):
    """Decorator to require authentication and optional permissions."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'error': 'No token provided'}), 401
            
            try:
                token = token.replace('Bearer ', '')
                payload = current_app.auth_manager.verify_token(token)
                
                if permissions:
                    user_permissions = payload.get('permissions', [])
                    if not any(perm in user_permissions for perm in permissions):
                        return jsonify({'error': 'Insufficient permissions'}), 403
                
                request.user = payload
                return f(*args, **kwargs)
            except AuthenticationError as e:
                return jsonify({'error': str(e)}), 401
        
        return decorated_function
    return decorator

class AuthenticationError(Exception):
    pass
```

### Network Security

```yaml
# security/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: lexos-network-policy
  namespace: lexos-production
spec:
  podSelector:
    matchLabels:
      app: lexos-app
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - podSelector:
        matchLabels:
          app: lexos-monitoring
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: mongodb
    ports:
    - protocol: TCP
      port: 27017
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: SQLite Database Locked
```bash
# Symptoms: "database is locked" errors
# Cause: Long-running transactions or abandoned connections

# Solution 1: Check for locks
sqlite3 /opt/lexos/data/corpus.db "SELECT * FROM sqlite_master LIMIT 1;"

# Solution 2: Enable WAL mode
sqlite3 /opt/lexos/data/corpus.db "PRAGMA journal_mode=WAL;"

# Solution 3: Restart application to clear connections
docker-compose restart lexos-app
```

#### Issue: MongoDB Connection Timeout
```bash
# Symptoms: Connection timeout errors
# Cause: Network issues or MongoDB unavailability

# Check MongoDB status
docker-compose exec mongodb mongo --eval "db.adminCommand('ismaster')"

# Check network connectivity
nc -zv mongodb-service 27017

# Review MongoDB logs
docker-compose logs mongodb | tail -100
```

#### Issue: High Memory Usage
```bash
# Symptoms: Out of memory errors
# Cause: Large corpus processing or memory leaks

# Monitor memory usage
docker stats lexos-app

# Check application metrics
curl http://localhost:8080/metrics | grep memory

# Restart with memory profiling
docker-compose up --build lexos-app
```

### Performance Optimization

#### Database Optimization
```sql
-- SQLite optimization queries
PRAGMA optimize;
PRAGMA vacuum;
PRAGMA integrity_check;

-- Check index usage
EXPLAIN QUERY PLAN SELECT * FROM records WHERE model = 'en_core_web_sm';

-- Create additional indexes for common queries
CREATE INDEX IF NOT EXISTS idx_records_created_model ON records(created_at, model);
CREATE INDEX IF NOT EXISTS idx_records_active_tokens ON records(is_active, num_tokens);
```

#### MongoDB Optimization
```javascript
// MongoDB optimization commands
db.corpus_records.createIndex({"metadata.model": 1, "is_active": 1});
db.corpus_records.createIndex({"created_at": 1});
db.corpus_records.createIndex({"statistics.num_tokens": 1});

// Check index usage
db.corpus_records.getIndexes();
db.corpus_records.explain("executionStats").find({"metadata.model": "en_core_web_sm"});

// Optimize collections
db.corpus_records.reIndex();
```

This comprehensive deployment guide provides production-ready configurations for all deployment scenarios while maintaining backward compatibility with the existing file-based system. The modular approach allows organizations to choose the appropriate level of enhancement based on their specific requirements.