resource "aws_db_parameter_group" "pg14_autovacuum" {
  name   = "pg14-autovacuum-optimized"
  family = "postgres14"

  # Enable autovacuum
  parameter {
    name  = "autovacuum"
    value = "on"
  }

  # More workers for parallel cleanup
  parameter {
    name  = "autovacuum_max_workers"
    value = "5"
  }

  # Start vacuum when 5% of table changes
  parameter {
    name  = "autovacuum_vacuum_scale_factor"
    value = "0.05"
  }

  # Start analyze when 2% changes
  parameter {
    name  = "autovacuum_analyze_scale_factor"
    value = "0.02"
  }

  # Allow faster cleanup
  parameter {
    name  = "autovacuum_vacuum_cost_limit"
    value = "2000"
  }

  # Reduce delay between cleanup operations
  parameter {
    name  = "autovacuum_vacuum_cost_delay"
    value = "10"
  }
}
