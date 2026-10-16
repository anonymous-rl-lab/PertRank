#!/usr/bin/env Rscript
# Native ashr posterior means; alternatively official immutable TRADE fit_ash.
# Binary matrices are little-endian float64, target-major, full declared gene axis.
# No benchmark scores or reference expressions are read here.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=3L) stop("usage: posterior_runner.R input_dir output_dir config.json")
suppressPackageStartupMessages(library(ashr))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(packageDescription("ashr")$Version == "2.2-63")
inp <- normalizePath(args[1]); out <- normalizePath(args[2]); cfg <- fromJSON(args[3])
meta <- fromJSON(file.path(inp, "metadata.json"))
n <- as.integer(meta$n_rows); p <- as.integer(meta$n_genes)
stopifnot(n>0L,p>1L, cfg$mode %in% c("ash", "trade"))
ids <- readLines(file.path(inp,"row_ids.txt"),warn=FALSE)
genes <- readLines(file.path(inp,"genes.txt"),warn=FALSE)
stopifnot(length(ids)==n,length(genes)==p,!anyDuplicated(ids),!anyDuplicated(genes))
if(cfg$mode=="trade") source(cfg$trade_source,local=globalenv())
ef <- file(file.path(inp,"effects.f64"),"rb")
sf <- file(file.path(inp,"ses.f64"),"rb")
pf <- file(file.path(out,"posterior.f64"),"wb")
report <- file(file.path(out,"fit_rows.jsonl"),"wt")
set.seed(as.integer(cfg$seed))
start_all <- proc.time()[3]
failed <- 0L; warn_count <- 0L
for(i in seq_len(n)) {
  y <- readBin(ef,"double",n=p,size=8,endian="little")
  s <- readBin(sf,"double",n=p,size=8,endian="little")
  stopifnot(length(y)==p,length(s)==p,all(is.finite(y)),all(is.finite(s)),all(s>0))
  warnings <- character(); start <- proc.time()[3]
  fit <- tryCatch(withCallingHandlers({
    if(cfg$mode=="trade") {
      if(min(y)==max(y)) stop("degenerate TRADE effect range")
      # Unmodified official function; no filtering, jitter or optimizer fallback.
      fit_ash(y,s,min(y),max(y),n_sample=0)$fit
    } else {
      # The explicit config selects only documented native ashr options.
      do.call(ashr::ash, c(list(betahat=y,sebetahat=s,outputlevel=c("fitted_g","PosteriorMean","fit_details","loglik")), cfg$ash_options))
    }
  },warning=function(w) {
    warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")
  }),error=function(e)e)
  primary_error <- if(inherits(fit,"error")) conditionMessage(fit) else NULL
  primary_warning <- warnings
  primary_converged <- if(inherits(fit,"error")) FALSE else fit$fit_details$optreturn$converged
  primary_seconds <- unname(proc.time()[3]-start)
  primary_solver <- if(cfg$mode=="trade" || is.null(cfg$ash_options$optmethod)) "mixSQP" else cfg$ash_options$optmethod
  used_solver <- primary_solver; retried <- FALSE
  primary_info <- if(inherits(fit,"error")) NULL else fit$fit_details$optreturn
  if(!is.null(primary_info)) saveRDS(list(fitted_g=fit$fitted_g,loglik=fit$loglik,
    optreturn=primary_info,posterior=ashr::get_pm(fit)),file.path(out,sprintf("primary_%06d.rds",i)))
  if(isTRUE(cfg$retry_mixem) && primary_solver=="mixSQP" &&
     (inherits(fit,"error") || identical(primary_converged,FALSE))) {
    retried <- TRUE;used_solver <- "mixEM";warnings <- character()
    fit <- tryCatch(withCallingHandlers({
      if(cfg$mode=="trade") {
        # Explicit numerical retry only, retaining the official TRADE model/grid.
        ashr::ash(y,s,mixcompdist="halfuniform",outputlevel=3,
                  grange=c(min(y),max(y)),prior="uniform",optmethod="mixEM",
                  control=list(maxiter=10000))
      } else {
        retryopts <- cfg$ash_options;retryopts$optmethod <- "mixEM"
        retryopts$control <- list(maxiter=10000)
        do.call(ashr::ash,c(list(betahat=y,sebetahat=s,outputlevel=c("fitted_g","PosteriorMean","fit_details","loglik")),retryopts))
      }
    },warning=function(w) {
      warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")
    }),error=function(e)e)
  }
  final_converged <- if(inherits(fit,"error")) FALSE else fit$fit_details$optreturn$converged
  if(!inherits(fit,"error") && !isTRUE(final_converged)) {
    saveRDS(list(fitted_g=fit$fitted_g,loglik=fit$loglik,
      optreturn=fit$fit_details$optreturn,posterior=ashr::get_pm(fit)),
      file.path(out,sprintf("unconverged_%06d.rds",i)))
    fit <- simpleError("Native optimizer failed convergence criterion")
  }
  seconds <- unname(proc.time()[3]-start)
  if(inherits(fit,"error")) {
    failed <- failed+1L; pm <- rep(NaN,p)
    rec <- list(row=i,row_id=ids[i],status="FAILED",error=conditionMessage(fit),
                warnings=warnings,seconds=seconds)
  } else {
    pm <- ashr::get_pm(fit)
    if(length(pm)!=p || any(!is.finite(pm))) stop("nonfinite or wrong-size posterior means")
    compact <- list(fitted_g=fit$fitted_g,loglik=fit$loglik,
                    penloglik=fit$fit_details$penloglik,
                    optmethod=fit$fit_details$optmethod,optreturn=fit$fit_details$optreturn)
    saveRDS(compact,file.path(out,sprintf("fit_%06d.rds",i)))
    # Compact mixture and optimizer fields preserve numerical auditability.
    rec <- list(row=i,row_id=ids[i],status="OK",warnings=warnings,seconds=seconds,
                loglik=unname(fit$loglik),n_components=length(fit$fitted_g$pi),
                fitted_g=unclass(fit$fitted_g),optreturn=fit$fit_details$optreturn)
  }
  rec$primary_error <- primary_error;rec$primary_warning <- primary_warning
  rec$primary_converged <- primary_converged;rec$primary_seconds <- primary_seconds
  rec$primary_solver <- primary_solver;rec$used_solver <- used_solver
  rec$retry_mixem <- retried;rec$final_converged <- final_converged
  warn_count <- warn_count+length(warnings)
  writeBin(as.double(pm),pf,size=8,endian="little")
  writeLines(toJSON(rec,auto_unbox=TRUE,null="null",digits=16),report); flush(report); flush(pf)
  cat(sprintf("ROW %d/%d %s %.3fs\n",i,n,rec$status,seconds)); flush.console()
}
stopifnot(length(readBin(ef,"double",n=1L,size=8,endian="little"))==0L,
          length(readBin(sf,"double",n=1L,size=8,endian="little"))==0L)
close(ef);close(sf);close(pf);close(report)
runtime <- list(status=if(failed==0L) "COMPLETE" else "INCOMPLETE_FAILED_FITS",
  config=cfg,full_declared_gene_axis=TRUE,n_rows=n,n_genes=p,failed_rows=failed,
  warnings=warn_count,seconds=unname(proc.time()[3]-start_all),
  R=R.version.string,ashr=packageDescription("ashr")$Version,
  effect_scale=meta$effect_scale,session=capture.output(sessionInfo()),
  read_benchmark_reference=FALSE,score_computed=FALSE)
write_json(runtime,file.path(out,"runtime.json"),pretty=TRUE,auto_unbox=TRUE,digits=16)
if(failed>0L) quit(status=2L)
