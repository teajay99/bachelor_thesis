#!/usr/bin/env Rscript

library(hadron)

args = commandArgs(trailingOnly = TRUE)

read_file <- function(path, thermTime) {
  csvfile = read.csv(path, sep = "\t", header = FALSE)
  if (ncol(csvfile) == 1) {
    csvfile = read.csv(path, sep = " ", header = FALSE)
  }
  data = csvfile[, c("V2")]
  data = data[-(1:thermTime)]
  out <- tryCatch({
    uwerrprimary(data)
  }, error = function(cond) {
    return(list(value = mean(data), dvalue = 0)
)
  })
  return(list(val = out$value, err = out$dvalue))
}

evaluate <- function(dataDirPath, thermTime, N, outFilePath) {

  out = matrix(NA, nrow = N, ncol = 3)

  for (i in 0:(N - 1)) {
    line = read_file(paste(dataDirPath, "/data-", i, ".csv", sep = ""), thermTime)
    out[i + 1, 1] = i
    out[i + 1, 2] = line$val
    out[i + 1, 3] = line$err
  }
  write.table(out, file = outFilePath, sep = "\t", row.names = FALSE, col.names = FALSE)
}

if (length(args) != 4) {
  print("4 Args required: data/directory/path #thermalizationIterations #Datasets output/file.csv")
  q()
}

evaluate(args[1], as.integer(args[2]), as.integer(args[3]), args[4])
