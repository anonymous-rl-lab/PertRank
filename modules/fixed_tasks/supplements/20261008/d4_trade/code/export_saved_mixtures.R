#!/usr/bin/env Rscript
# Export saved fitted distributions only; this does not perform fitting.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
rows <- list()
for (d in sort(list.dirs(args[1],full.names=TRUE,recursive=FALSE))) {
  if (!file.exists(file.path(d,'COMPLETE.json'))) next
  fits <- readRDS(file.path(d,'result_mixtures.rds'))
  tab <- read.delim(file.path(d,'result_weights.tsv'),check.names=FALSE)
  for (i in seq_len(nrow(tab))) {
    f <- fits[[paste(tab$target[i],tab$replicate[i],sep='::')]]
    rows[[length(rows)+1L]] <- list(target=tab$target[i],replicate=tab$replicate[i],
      pi=f$fitted_g$pi,a=f$fitted_g$a,b=f$fitted_g$b,loglik=f$loglik)
  }
}
stopifnot(length(rows)==2945L)
jsonlite::write_json(rows,args[2],auto_unbox=TRUE,digits=NA)
