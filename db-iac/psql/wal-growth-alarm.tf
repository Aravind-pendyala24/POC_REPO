resource "aws_cloudwatch_metric_alarm" "wal_growth_alarm" {
  alarm_name          = "rds-wal-growth-high"
  namespace           = "AWS/RDS"
  metric_name         = "TransactionLogsDiskUsage"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 2
  comparison_operator = "GreaterThanThreshold"

  # Example: 20 GB threshold
  threshold = 20 * 1024 * 1024 * 1024

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgres.id
  }

  alarm_description = "WAL disk usage exceeded safe threshold"
}


resource "aws_cloudwatch_metric_alarm" "free_storage_alarm" {
  alarm_name          = "rds-free-storage-low"
  namespace           = "AWS/RDS"
  metric_name         = "FreeStorageSpace"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 2
  comparison_operator = "LessThanThreshold"

  # 100 GB free threshold
  threshold = 100 * 1024 * 1024 * 1024

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgres.id
  }

  alarm_description = "Free storage space below 100GB"
}
