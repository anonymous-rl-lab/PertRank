#!/usr/bin/env Rscript
# Run under R 4.4.3 with a working C/C++/Fortran toolchain.
args <- commandArgs(trailingOnly=TRUE)
lib <- if(length(args)) args[1] else file.path(getwd(), 'R-library')
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(normalizePath(lib), .libPaths()))
stopifnot(as.character(getRversion()) == '4.4.3')
versions <- c(lattice='0.23-1', Matrix='1.7-6', Rcpp='1.1.2', RcppArmadillo='15.6.0-1',
 SQUAREM='2026.1', truncnorm='1.0-9', etrunct='0.1', invgamma='1.2',
 irlba='2.4.1', mixsqp='0.3-54', jsonlite='2.0.0', ashr='2.2-63')
for (pkg in names(versions)) {
 wanted <- versions[[pkg]]
 present <- tryCatch(unname(installed.packages()[pkg,'Version']), error=function(e) '')
 if(present == wanted) next
 name <- paste0(pkg, '_', wanted, '.tar.gz')
 urls <- c(paste0('https://cran.r-project.org/src/contrib/', name),
           paste0('https://cran.r-project.org/src/contrib/Archive/', pkg, '/', name))
 dst <- tempfile(fileext='.tar.gz'); success <- FALSE
 for(url in urls) {
  ok <- tryCatch({download.file(url, dst, mode='wb', quiet=FALSE); TRUE}, error=function(e) FALSE)
  if(ok) {success <- TRUE; break}
 }
 if(!success) stop(paste('Cannot download fixed package', pkg, wanted))
 install.packages(dst, lib=lib, repos=NULL, type='source')
 stopifnot(unname(installed.packages()[pkg,'Version']) == wanted)
 unlink(dst)
}
library(ashr);library(jsonlite)
print(installed.packages()[names(versions),'Version'])
cat('FIXED R ENVIRONMENT READY\n')
