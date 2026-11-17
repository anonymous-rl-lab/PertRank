#!/usr/bin/env Rscript
# Native CRAN rvalues 0.7.1, one prespecified Gaussian-conjugate configuration.
# This program reads only candidate-level shallow effects and marginal SEs.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) stop('Usage: run_rvalues.R INPUT.tsv OUTPUT.tsv')
input <- args[1]; output <- args[2]
library(rvalues)
stopifnot(as.character(packageVersion('rvalues')) == '0.7.1')
set.seed(20261116)
dat <- read.delim(input, sep='\t', header=TRUE, check.names=FALSE,
                  stringsAsFactors=FALSE, quote='', comment.char='')
req <- c('task','query_index','query_id','direction','target_index','target_id','effect','se')
if (!all(req %in% names(dat))) stop('Missing required input column')
if (!('input_id' %in% names(dat))) dat$input_id <- '0'
if (!all(dat$direction %in% c(-1,1))) stop('direction must equal -1 or +1')
dat$score <- NA_real_; dat$rvalue <- NA_real_; dat$fit_status <- NA_character_
keys <- paste(dat$task, dat$input_id, dat$query_index, dat$direction, sep='|')
keylist <- unique(keys)
diag <- vector('list', length(keylist))
hyper <- getFromNamespace('HyperEstimate','rvalues')
for (ii in seq_along(keylist)) {
  k <- keylist[ii]; ix <- which(keys == k)
  d <- dat[ix,,drop=FALSE]
  xx <- as.numeric(d$direction) * as.numeric(d$effect)
  ss <- as.numeric(d$se)
  warnings <- character(); err <- ''; status <- 'failed'
  mu <- NA_real_; tau2 <- NA_real_; grid_n <- NA_integer_
  rv <- rep(NA_real_, length(ix)); score <- rv
  tryCatch(withCallingHandlers({
    if (length(ix)<3 || anyDuplicated(d$target_id)) stop('Invalid candidate count or duplicate target')
    if (any(!is.finite(xx)) || any(!is.finite(ss)) || any(ss<=0)) stop('Effects must be finite; SE must be finite and strictly positive')
    hh <- hyper(xx, ss, family='gaussian'); mu <- hh[1]; tau2 <- hh[2]
    if (!is.finite(mu) || !is.finite(tau2) || tau2<0) stop('Invalid fitted prior parameters')
    if (tau2 == 0) {
      # Exact boundary of the native fitted model: a common point mass.
      # The original implementation divides by tau2; emit an uninformative tie,
      # not a manufactured native r-value, and identify this adaptation explicitly.
      score <- rep(0,length(ix)); status <- 'boundary_degenerate'
    } else {
      fit <- rvalues(cbind(xx,ss), family=gaussian, prior='conjugate',
                     hypers='estimate', alpha.grid=NULL, ngrid=NULL,
                     smooth='none', control=list())
      rv <- as.numeric(fit$rvalues)
      if (length(rv) != length(ix) || any(!is.finite(rv))) stop('Nonfinite or mismatched native r-values')
      if (any(rv<0 | rv>1)) stop('Native r-values outside [0,1]')
      if (!isTRUE(all.equal(unname(fit$aux$hypers),unname(hh),tolerance=1e-12))) stop('Prior estimation is not deterministic')
      score <- -rv
      grid_n <- length(fit$aux$alpha.grid)
      status <- 'native_success'
    }
  }, warning=function(w) { warnings <<- c(warnings,conditionMessage(w)); invokeRestart('muffleWarning') }),
    error=function(e) {err <<- conditionMessage(e)})
  dat$score[ix] <- score; dat$rvalue[ix] <- rv; dat$fit_status[ix] <- status
  diag[[ii]] <- data.frame(task=d$task[1],input_id=d$input_id[1],query_index=d$query_index[1],
       query_id=d$query_id[1],direction=d$direction[1],n_candidates=length(ix),
       status=status,prior_mean=mu,prior_variance=tau2,alpha_grid_n=grid_n,
       n_unique_scores=length(unique(score)),warnings=paste(unique(warnings),collapse=' | '),
       error=err,stringsAsFactors=FALSE)
  if(ii %% 100 == 0) cat(sprintf('Completed %d/%d fits\n',ii,length(keylist)))
}
# Preserve native binary64 values and emit enough decimal digits for exact round-trip.
saveRDS(dat,paste0(output,'.native.rds'),version=3)
export_dat <- dat
for (nm in c('score','rvalue')) {
  export_dat[[nm]] <- ifelse(is.na(dat[[nm]]),NA_character_,sprintf('%.17g',dat[[nm]]))
}
write.table(export_dat,output,sep='\t',row.names=FALSE,quote=FALSE,na='NA')
dd <- do.call(rbind,diag)
write.table(dd,paste0(output,'.fits.tsv'),sep='\t',row.names=FALSE,quote=FALSE,na='NA')
capture.output(sessionInfo(),file=paste0(output,'.sessionInfo.txt'))
print(table(dd$status))
if (any(dd$status=='failed')) quit(status=2)
