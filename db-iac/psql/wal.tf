resource "aws_db_parameter_group" "pg14_wal" {
  name   = "pg14-wal-cleanup"
  family = "postgres14"

  # Maximum WAL before forced checkpoint
  parameter {
    name  = "max_wal_size"
    value = "20480"   # MB = 20 GB
  }

  # Minimum WAL retained
  parameter {
    name  = "min_wal_size"
    value = "1024"    # 1 GB
  }

  # Time-based checkpoint
  parameter {
    name  = "checkpoint_timeout"
    value = "900"     # 15 minutes
  }

  # Required for replicas
  parameter {
    name  = "wal_level"
    value = "replica"
  }
}
