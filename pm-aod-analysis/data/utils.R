# Some helper functions

read.csv.maker <- function(fp, ...) {
  function(file, ...) {
    read.csv(file.path(fp, file), stringsAsFactors = FALSE, ...)
  }
}

write.csv.maker <- function(fp) {
  function(..., file) {
    write.csv(..., file = file.path(fp, file), row.names = FALSE, quote = FALSE)
  }
}