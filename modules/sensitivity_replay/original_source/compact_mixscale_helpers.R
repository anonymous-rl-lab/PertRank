# Storage-only helpers. No normalization, PCA, neighbor search, scoring or fitting.
# Native Mixscale calls stay in the unmodified official package.
suppressPackageStartupMessages({library(Seurat);library(Matrix);library(jsonlite)})

compact_assay_spec <- function(assay, ncell) {
  stopifnot(inherits(assay, 'Assay'), !inherits(assay, 'Assay5'))
  dynamic <- character(); static <- list()
  for (nm in slotNames(assay)) {
    value <- slot(assay, nm)
    ismat <- is.matrix(value) || inherits(value, 'Matrix')
    if (ismat && ncol(value) == ncell) dynamic <- c(dynamic, nm)
    else static[nm] <- list(value)
  }
  list(class=class(assay)[[1]], dynamic=dynamic, static=static)
}

compact_take <- function(object, cells, specs, target_group=NULL, index=NULL) {
  stopifnot(!anyNA(match(cells,colnames(object))), !anyDuplicated(cells))
  mats <- lapply(names(specs), function(aname) {
    ans <- lapply(specs[[aname]]$dynamic, function(sn) slot(object[[aname]],sn)[,cells,drop=FALSE])
    names(ans) <- specs[[aname]]$dynamic; ans
  }); names(mats) <- names(specs)
  list(schema='mixscale-lossless-block-v1', cells=cells,
       global_cell_index=match(cells,colnames(object)),
       metadata=object@meta.data[cells,,drop=FALSE],
       identities=object@active.ident[cells], matrices=mats,
       target_group=target_group, cohort_position=index)
}

load_compact <- function(shared, target_files) {
  if (is.character(shared)) {stopifnot(length(shared)==1); shared <- readRDS(shared)}
  if (is.character(target_files)) targets <- lapply(target_files,readRDS)
  else if (is.list(target_files) && identical(target_files$schema,'mixscale-lossless-block-v1')) targets <- list(target_files)
  else targets <- target_files
  stopifnot(length(targets)>0, identical(shared$schema,'mixscale-lossless-shared-v1'))
  stopifnot(all(vapply(targets,function(x) identical(x$schema,'mixscale-lossless-block-v1'),logical(1))))
  blocks <- c(list(shared$ntc),targets)
  target_cells <- unlist(lapply(targets,`[[`,'cells'),use.names=FALSE)
  target_indices <- unlist(lapply(targets,`[[`,'global_cell_index'),use.names=FALSE)
  stopifnot(!anyDuplicated(c(shared$ntc$cells,target_cells)),!anyDuplicated(target_indices))
  # Preserve original NTC order; then retain selected target cells' global order.
  original_cells <- unlist(lapply(blocks,`[[`,'cells'),use.names=FALSE)
  cells <- c(shared$ntc$cells,target_cells[order(target_indices)])
  ordering <- match(cells,original_cells); stopifnot(!anyNA(ordering))
  assays <- lapply(names(shared$specs),function(aname) {
    spec <- shared$specs[[aname]]; args <- spec$static
    for (sn in spec$dynamic) {
      values <- lapply(blocks,function(b)b$matrices[[aname]][[sn]])
      stopifnot(all(vapply(values,function(x)identical(rownames(x),rownames(values[[1]])),logical(1))))
      args[sn] <- list(do.call(cbind,values)[,ordering,drop=FALSE])
    }
    ans <- do.call(methods::new,c(list(Class=spec$class),args))
    methods::validObject(ans); ans
  }); names(assays) <- names(shared$specs)
  md <- do.call(rbind,lapply(blocks,`[[`,'metadata'))[cells,,drop=FALSE]
  options(Seurat.object.assay.version='v3')
  obj <- CreateSeuratObject(counts=assays[['RNA']]@counts,assay='RNA',meta.data=md,project=shared$project)
  obj@assays <- assays
  obj@meta.data <- md
  iv <- unlist(lapply(blocks,function(b)as.character(b$identities)),use.names=FALSE)[ordering]
  obj@active.ident <- factor(iv,levels=shared$identity_levels);names(obj@active.ident)<-cells
  obj@tools <- shared$tools; obj@commands <- shared$commands
  obj@misc <- shared$misc;obj@version <- shared$version
  DefaultAssay(obj) <- 'RNA'
  stopifnot(identical(colnames(obj),cells),identical(rownames(obj@meta.data),cells))
  methods::validObject(obj);obj
}

check_compact_identity <- function(original, reconstructed, shared) {
  cells <- colnames(reconstructed);checks <- list()
  for (aname in names(shared$specs)) {
    spec <- shared$specs[[aname]]
    for (sn in slotNames(original[[aname]])) {
      ref <- slot(original[[aname]],sn)
      if (sn %in% spec$dynamic) ref <- ref[,cells,drop=FALSE]
      checks[[paste(aname,sn,sep='.')]] <- identical(ref,slot(reconstructed[[aname]],sn))
    }
  }
  checks$metadata <- identical(original@meta.data[cells,,drop=FALSE],reconstructed@meta.data)
  checks$identities <- identical(original@active.ident[cells],reconstructed@active.ident)
  checks$tools <- identical(original@tools,reconstructed@tools)
  checks$commands <- identical(original@commands,reconstructed@commands)
  checks$ntc_order <- identical(cells[seq_along(shared$ntc$cells)],shared$ntc$cells)
  stopifnot(all(unlist(checks)));checks
}

export_compact <- function(global_rds, input_dir, out_dir, indices=NULL, check_every=1L) {
  dir.create(out_dir,recursive=TRUE,showWarnings=FALSE)
  t0 <- proc.time()[['elapsed']]
  obj <- readRDS(global_rds);stopifnot(all(c('RNA','PRTB') %in% names(obj@assays)))
  tg <- read.csv(file.path(input_dir,'targets.csv'),stringsAsFactors=FALSE)
  stopifnot('target_group' %in% names(tg), !anyDuplicated(tg$target_group))
  if (is.null(indices)) indices <- seq_len(nrow(tg))
  stopifnot(all(indices>=1 & indices<=nrow(tg)),!anyDuplicated(indices))
  md <- obj@meta.data; ntc <- rownames(md)[as.character(md$target_group)=='NT']
  stopifnot(length(ntc)==2000,!anyDuplicated(ntc))
  specs <- lapply(obj@assays[c('RNA','PRTB')],compact_assay_spec,ncell=ncol(obj))
  shared <- list(schema='mixscale-lossless-shared-v1',specs=specs,
      ntc=compact_take(obj,ntc,specs,target_group='NT'),
      tools=obj@tools, commands=obj@commands, misc=obj@misc, version=obj@version,
      project=obj@project.name,identity_levels=levels(obj@active.ident),
      source=list(path=normalizePath(global_rds),cells=ncol(obj),assays=c('RNA','PRTB'),
                  identity='Lossless execution representation of the unchanged global preprocessing; not an alternative fit.'))
  saveRDS(shared,file.path(out_dir,'shared_ntc.rds'),compress=FALSE)
  saved_shared <- readRDS(file.path(out_dir,'shared_ntc.rds'))
  stopifnot(identical(shared,saved_shared));rm(saved_shared);invisible(gc())
  manifest <- list(); checked <- list()
  for (ii in indices) {
    label <- as.character(tg$target_group[[ii]])
    cells <- rownames(md)[as.character(md$target_group)==label]
    stopifnot(length(cells)==20)
    block <- compact_take(obj,cells,specs,target_group=label,index=ii)
    path <- file.path(out_dir,sprintf('target_%04d.rds',ii))
    saveRDS(block,path,compress=FALSE)
    saved_block <- readRDS(path)
    stopifnot(identical(block,saved_block));rm(saved_block)
    # Check values read back from disk, not only in-memory serialized inputs.
    if (check_every>0L && ((match(ii,indices)-1L) %% check_every==0L || ii==tail(indices,1))) {
      cc <- load_compact(shared,path)
      checked[[as.character(ii)]] <- check_compact_identity(obj,cc,shared)
      rm(cc);invisible(gc())
    }
    manifest[[length(manifest)+1L]] <- data.frame(cohort_position=ii,target_group=label,
        file=basename(path),cells=length(cells),bytes=file.info(path)$size,
        stringsAsFactors=FALSE)
    if(length(manifest) %% 20L==0L) {cat(length(manifest),'of',length(indices),'exported\n');flush.console()}
  }
  write.csv(do.call(rbind,manifest),file.path(out_dir,'compact_manifest.csv'),row.names=FALSE)
  write_json(list(status='ALL_SERIALIZED_BLOCKS_EXACT_AND_RECONSTRUCTED_SLOT_IDENTITY_VERIFIED',targets=length(indices),
    exact_disk_roundtrip_blocks=length(indices),shared_exact_disk_roundtrip=TRUE,
    compact_reconstruction_checks=length(checked),
    source_cells=ncol(obj),ntc_cells=length(ntc),checked_target_slots=checked,
    shared_bytes=file.info(file.path(out_dir,'shared_ntc.rds'))$size,
    seconds=proc.time()[['elapsed']]-t0,
    omitted_slots='Global PCA/reductions and graphs are not needed after CalcPerturbSig by the fixed native calls. Tools and commands are retained unmodified.'),
    file.path(out_dir,'COMPACT_EXPORT_IDENTITY.json'),auto_unbox=TRUE,pretty=TRUE)
  invisible(manifest)
}
