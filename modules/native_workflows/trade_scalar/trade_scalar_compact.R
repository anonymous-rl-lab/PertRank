#!/usr/bin/env Rscript
# TRADE-core common-coordinate scalar adapter; not native DESeq2 TRADE.
# Run: Rscript trade_scalar_compact.R effects.tsv output_prefix [official_source.R]
# effects.tsv: target, replicate, gene, effect, se. Fixed complete gene axis.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2L) stop("usage: trade_scalar_compact.R effects.tsv output_prefix [official_source.R]")
suppressPackageStartupMessages(library(ashr))
suppressPackageStartupMessages(library(jsonlite))
src <- if(length(args)>=3L) args[3] else file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1]))), "upstream/R_TRADE_univariate.R")
source(src, local=globalenv())
input <- read.delim(args[1], check.names=FALSE, stringsAsFactors=FALSE)
required <- c("target", "replicate", "gene", "effect", "se")
stopifnot(all(required %in% names(input)), nrow(input)>0L)
stopifnot(all(is.finite(input$effect)), all(is.finite(input$se)), all(input$se>0))
keys <- unique(input[c("target", "replicate")])
expected_genes <- NULL
rows <- list(); fits <- list()
set.seed(2026)
for(i in seq_len(nrow(keys))) {
  d <- input[input$target==keys$target[i] & input$replicate==keys$replicate[i], ]
  stopifnot(!anyDuplicated(d$gene))
  if(is.null(expected_genes)) expected_genes <- d$gene
  stopifnot(identical(d$gene, expected_genes), length(d$gene)>1L)
  # Source the immutable official fit_ash(), which calls ashr::ash with
  # halfuniform components, uniform prior, and its unmodified adaptive grid.
  # n_sample=0 skips Monte Carlo output; moments below are exact analytically.
  if(min(d$effect)==max(d$effect)) stop("constant effect vector: fit range degenerates; no silent fallback")
  start <- proc.time()[3]
  z <- fit_ash(d$effect, d$se, min(d$effect), max(d$effect), n_sample=0)
  g <- z$fit$fitted_g
  mu_components <- (g$a+g$b)/2
  var_components <- (g$b-g$a)^2/12
  mu <- sum(g$pi*mu_components)
  TI <- sum(g$pi*(var_components+(mu_components-mu)^2))
  S <- TI+mu^2
  S_direct <- sum(g$pi*(g$a^2+g$a*g$b+g$b^2)/3)
  stopifnot(abs(sum(g$pi)-1)<1e-6, is.finite(S), S>=0,
            isTRUE(all.equal(S, S_direct, tolerance=1e-10)))
  nu <- mean(d$se^2)
  w <- S/(S+nu)
  stopifnot(is.finite(w), w>=0, w<=1)
  rows[[i]] <- data.frame(target=keys$target[i], replicate=keys$replicate[i],
    n_genes=nrow(d), latent_mean=mu, TI=TI, S=S, nu=nu, weight=w,
    n_components=length(g$pi), loglik=z$fit$loglik,
    seconds=unname(proc.time()[3]-start))
  fits[[paste(keys$target[i],keys$replicate[i],sep="::")]] <- list(fitted_g=g, loglik=z$fit$loglik)
}
write.table(do.call(rbind,rows),paste0(args[2],"_weights.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
saveRDS(fits,paste0(args[2],"_mixtures.rds"))
stopifnot(identical(fits, readRDS(paste0(args[2],"_mixtures.rds"))))
write_json(list(adapter="TRADE-core common-coordinate scalar adapter",
  upstream_commit="3432d108c43325e9df65276563d65144a26b69ce",
  serialization="compact fitted_g and loglik; no per-gene intermediate likelihood matrix", source=normalizePath(src), R=R.version.string, ashr=as.character(packageVersion("ashr")),
  jsonlite=as.character(packageVersion("jsonlite")), seed=2026,
  genes_fixed=TRUE, native_DESeq2=FALSE, effect_scale="original normalized expression mean difference",
  component_family="halfuniform", prior="uniform", monte_carlo_samples=0,
  formula="S=TI+latent_mean^2; nu=mean(se^2); w=S/(S+nu); prediction=w*effect",
  installed_packages=as.data.frame(installed.packages()[,c("Package","Version")])),
  paste0(args[2],"_runtime.json"),pretty=TRUE,auto_unbox=TRUE)
cat("TRADE ADAPTER COMPLETED\n")
