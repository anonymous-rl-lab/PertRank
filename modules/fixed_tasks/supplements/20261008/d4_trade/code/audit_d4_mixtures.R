#!/usr/bin/env Rscript
# Read-only moment reconstruction from saved mixture objects; never refits.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
dirs <- list.dirs(args[1], full.names=TRUE, recursive=FALSE)
report <- list(); mx <- c(mean=0, TI=0, S=0, weight=0, loglik=0)
fits_checked <- 0L
for (d in dirs) {
  if (!file.exists(file.path(d, "COMPLETE.json"))) next
  f <- readRDS(file.path(d, "result_mixtures.rds"))
  tab <- read.delim(file.path(d, "result_weights.tsv"), check.names=FALSE)
  stopifnot(length(f)==nrow(tab))
  for (i in seq_len(nrow(tab))) {
    key <- paste(tab$target[i],tab$replicate[i],sep="::")
    fit <- f[[key]]
    stopifnot(!is.null(fit))
    g <- fit$fitted_g
    stopifnot(all(is.finite(g$pi)),all(g$pi>=0),abs(sum(g$pi)-1)<1e-6,
              all(is.finite(g$a)),all(is.finite(g$b)),all(g$a<=g$b))
    # Two-point Gauss-Legendre integrates each component's first and second
    # moments exactly, independently of the adapter's analytic variance code.
    lo <- (g$a+g$b)/2 - (g$b-g$a)/(2*sqrt(3))
    hi <- (g$a+g$b)/2 + (g$b-g$a)/(2*sqrt(3))
    mu <- sum(g$pi * (lo+hi)/2)
    s <- sum(g$pi * (lo^2+hi^2)/2)
    ti <- s-mu^2
    w <- s/(s+tab$nu[i])
    errs <- c(mean=abs(mu-tab$latent_mean[i]), TI=abs(ti-tab$TI[i]),
              S=abs(s-tab$S[i]), weight=abs(w-tab$weight[i]),
              loglik=abs(fit$loglik-tab$loglik[i]))
    stopifnot(all(is.finite(errs)),max(errs[1:4])<1e-10,errs[5]<1e-7,
              length(g$pi)==tab$n_components[i])
    mx <- pmax(mx,errs)
  }
  fits_checked <- fits_checked+nrow(tab)
  report[[length(report)+1L]] <- list(batch=basename(d),fits=nrow(tab))
}
out <- list(status="PASS_FOR_COMPLETED_MIXTURES",fits_checked=fits_checked,
            complete_formal_result=identical(fits_checked,2945L),
            maximum_absolute_errors=as.list(mx),batches=report,
            reconstruction="two-point Gauss-Legendre quadrature over each saved uniform component",
            refitting_performed=FALSE)
jsonlite::write_json(out,args[2],auto_unbox=TRUE,pretty=TRUE,digits=17)
cat(jsonlite::toJSON(out,auto_unbox=TRUE,pretty=TRUE,digits=17),"\n")
