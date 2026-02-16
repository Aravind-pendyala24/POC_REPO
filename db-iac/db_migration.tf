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
