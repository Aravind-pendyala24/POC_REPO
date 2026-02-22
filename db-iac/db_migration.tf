variable "sql_file" {
  type = string
}

resource "null_resource" "db_migration" {

  triggers = {
    sql_file      = var.sql_file
    sql_file_hash = filesha256("${path.module}/../sql/scripts/${var.sql_file}")
  }

  provisioner "local-exec" {
    command = <<EOT
      python3 ${path.module}/../scripts/run_migration.py \
        ${var.db_host} \
        ${var.db_name} \
        ${var.db_user} \
        ${var.db_password} \
        ${path.module}/../../sql/scripts/${var.sql_file}
    EOT
  }
}

#v2
resource "null_resource" "db_migration" {

  for_each = var.execute_migration ? { run = true } : {}

  triggers = {
    sql_file      = var.sql_file
    sql_file_hash = filesha256("${path.module}/../sql/scripts/${var.sql_file}")
  }

  provisioner "local-exec" {
    command = <<EOT
      python3 ${path.module}/../scripts/run_migration.py \
        ${var.environment} \
        ${var.db_host} \
        ${var.db_name} \
        ${var.db_user} \
        ${var.db_password} \
        ${path.module}/../sql/scripts/${var.sql_file}
    EOT
  }
}

#v3
resource "null_resource" "db_migration" {

  count = var.execute_migration ? 1 : 0

  triggers = {
    sql_files = join("|", var.sql_files)
  }

  provisioner "local-exec" {

    command = "python3 ${path.module}/../scripts/run_migration.py"

    environment = {
      ENVIRONMENT   = var.environment
      DB_HOST       = module.postgres.cluster_endpoint
      DB_PORT       = "5432"
      DB_NAME       = var.db_name
      DB_USER       = var.db_username
      DB_PASSWORD   = local.db_secret.password

      # Simple delimiter string
      SQL_FILES     = join("|", var.sql_files)

      SQL_BASE_PATH = "${path.module}/../sql/scripts"
    }
  }
}
