#!/usr/bin/env Rscript
args=commandArgs(trailingOnly=TRUE)
if(length(args)!=3) stop('usage: run_official_mixscale.R <input_dir> <output_dir> <smoke|full>')
in_dir=args[[1]];out_dir=args[[2]];mode=args[[3]];t_start=proc.time()[['elapsed']]
stopifnot(mode %in% c('smoke','full'))
dir.create(out_dir,recursive=TRUE,showWarnings=FALSE)
options(Seurat.object.assay.version='v3',future.globals.maxSize=6*1024^3)
suppressPackageStartupMessages({library(Seurat);library(Mixscale);library(Matrix);library(jsonlite)})
stopifnot(exists('TopDEGenesMixscape',asNamespace('Seurat')),exists('GetMissingPerturb',asNamespace('Seurat')))
set.seed(20261005)
memlog=list()
record_memory=function(stage,object=NULL){
  rss=readLines('/proc/self/status');rss=rss[grepl('^Vm(RSS|HWM):',rss)]
  m=list(stage=stage,linux_rss=rss)
  if(!is.null(object)){m$object_size_bytes=as.numeric(object.size(object));if('PRTB' %in% names(object@assays)){a=GetAssayData(object=object[['PRTB']],layer='data');m$PRTB_dimensions=dim(a);m$PRTB_class=class(a)}}
  memlog[[length(memlog)+1]]<<-m;write_json(memlog,file.path(out_dir,'memory_stages.json'),auto_unbox=TRUE,pretty=TRUE);cat('MEMORY',stage,paste(rss,collapse=';'),'\n');flush.console()
}
record_memory('packages_loaded')
meta=read.csv(file.path(in_dir,'cells.csv'),row.names=1,check.names=FALSE)
cm=Read10X_h5(file.path(in_dir,'raw_counts_10x.h5'),use.names=FALSE,unique.features=FALSE)
if(is.list(cm)) {stopifnot(length(cm)==1);cm=cm[[1]]}
stopifnot(identical(colnames(cm),rownames(meta)),!anyDuplicated(rownames(cm)),!anyDuplicated(gsub('_','-',rownames(cm),fixed=TRUE)))
stopifnot(all(Matrix::colSums(cm)==meta$total_counts))
if(mode=='smoke') {
  targets=read.csv(file.path(in_dir,'targets.csv'),stringsAsFactors=FALSE)
  pick=head(targets$target_group[order(-targets$euclidean_weight,targets$target_index)],2)
  idx=which(meta$target_group %in% c('NT',pick));cm=cm[,idx,drop=FALSE];meta=meta[idx,,drop=FALSE]
}
original_feature_ids=rownames(cm)
obj=CreateSeuratObject(counts=cm,meta.data=meta,assay='RNA',min.cells=0,min.features=0)
actual_feature_ids=rownames(obj[['RNA']])
stopifnot(identical(actual_feature_ids,gsub('_','-',original_feature_ids,fixed=TRUE)),!anyDuplicated(actual_feature_ids))
write.csv(data.frame(source_feature_id=original_feature_ids,actual_feature_id=actual_feature_ids),file.path(out_dir,'feature_name_map.csv'),row.names=FALSE)
rm(cm);invisible(gc());record_memory('RNA_created',obj)
# Keep true full-source library totals even if imported feature axes differ.
obj[['nCount_RNA']]=meta$total_counts
obj=NormalizeData(obj,normalization.method='LogNormalize',scale.factor=10000,verbose=FALSE)
invisible(gc());record_memory('after_NormalizeData',obj)
obj=FindVariableFeatures(obj,selection.method='vst',nfeatures=2000,verbose=FALSE)
invisible(gc());record_memory('after_FindVariableFeatures',obj)
obj=ScaleData(obj,verbose=FALSE)
invisible(gc());record_memory('after_ScaleData',obj)
obj=RunPCA(obj,npcs=50,seed.use=20261005,verbose=FALSE)
invisible(gc());record_memory('after_RunPCA',obj)
obj=CalcPerturbSig(object=obj,assay='RNA',slot='data',gd.class='target_group',nt.cell.class='NT',reduction='pca',ndims=40,num.neighbors=20,new.assay.name='PRTB',split.by=NULL)
record_memory('after_CalcPerturbSig',obj)
invisible(gc());t_mixstart=proc.time()[['elapsed']]
obj=RunMixscale(object=obj,assay='PRTB',slot='scale.data',labels='target_group',nt.class.name='NT',min.de.genes=5,min.cells=5,logfc.threshold=.2,de.assay='RNA',max.de.genes=100,new.class.name='mixscale_score',fine.mode=FALSE,harmonize=FALSE,pval.cutoff=.05,seed=10282021,verbose=FALSE,split.by=NULL)
invisible(gc());t_mixend=proc.time()[['elapsed']];record_memory('after_RunMixscale',obj)
score_tool=Tool(obj,slot='RunMixscale')
if(length(score_tool)==0) stop('No target has a usable score: native regression cannot construct the context; do not lower thresholds.')
qs=read.csv(file.path(in_dir,'queries.csv'),stringsAsFactors=FALSE)
# Seurat normalises underscores in feature names when constructing an Assay.
idxq=match(qs$feature_id,original_feature_ids);stopifnot(!anyNA(idxq))
query_ids=actual_feature_ids[idxq]
stopifnot(!anyDuplicated(query_ids))
all_targets=unique(meta$target_group[meta$target_group!='NT'])
if(mode=='full') {
  con=gzfile(file.path(out_dir,'mixscale_scored.rds'),open='wb',compression=1);saveRDS(obj,con);close(con)
  write_json(list(mode='full',source_commit='3d3a2cd12e62382df01c056536d967534b2de3a9',n_targets=length(all_targets),score_targets=names(score_tool),preprocessing_seconds=t_mixstart-t_start,mixscale_signature_seconds=t_mixend-t_mixstart,checkpoint_bytes=file.info(file.path(out_dir,'mixscale_scored.rds'))$size),file.path(out_dir,'SIGNATURE_COMPLETE.json'),auto_unbox=TRUE,pretty=TRUE)
}
status=list();times=numeric();t0=proc.time()[['elapsed']]
fit_one=function(i) {
  set.seed(20261005+i)
  tg=all_targets[[i]];tf=proc.time()[['elapsed']]
  rr=Run_wmvRegDE(object=obj,assay='RNA',slot='counts',labels='target_group',nt.class.name='NT',PRTB_list=tg,logfc.threshold=0,min.pct=.1,min.cells.group=10,total_ct_labels='nCount_RNA',split.by=NULL,verbose=FALSE,full.results=FALSE)
  stopifnot(length(rr)==1,tg %in% names(rr))
  r=rr[[tg]];stopifnot(all(c('gene_ID','beta_weight','DE_method') %in% names(r)),!anyDuplicated(r$gene_ID))
  keep=r$gene_ID %in% query_ids
  if(any(!is.finite(r$beta_weight[keep]))) stop(paste('nonfinite coefficient',tg))
  write.csv(r[keep,,drop=FALSE],file.path(out_dir,sprintf('target_%04d.csv',i)),row.names=FALSE)
  st=list(index=i,target_group=tg,n_query_returned=sum(keep),n_all_returned=nrow(r),fit_methods=unique(r$DE_method),seconds=proc.time()[['elapsed']]-tf)
  write_json(st,file.path(out_dir,sprintf('status_%04d.json',i)),pretty=TRUE,auto_unbox=TRUE)
  cat(mode,i,length(all_targets),tg,'seconds',proc.time()[['elapsed']]-tf,'\n');flush.console();return(st)
}
workers=if(mode=='smoke') 1L else as.integer(Sys.getenv('MIXSCALE_WORKERS','2'))
stopifnot(workers %in% c(1L,2L,4L))
if(workers==1L){status=lapply(seq_along(all_targets),fit_one)} else {status=parallel::mclapply(seq_along(all_targets),fit_one,mc.cores=workers,mc.preschedule=FALSE,mc.set.seed=FALSE)}
if(any(vapply(status,inherits,logical(1),'try-error'))) stop('An official target fit failed; no complete evaluation is permitted.')
write_json(status,file.path(out_dir,'target_status.json'),pretty=TRUE,auto_unbox=TRUE)
record_memory('regression_complete',obj)
writeLines(capture.output(sessionInfo()),file.path(out_dir,'sessionInfo.txt'))
write_json(list(mode=mode,workers=workers,source_commit='3d3a2cd12e62382df01c056536d967534b2de3a9',n_targets=length(all_targets),score_targets=names(score_tool),total_elapsed_seconds=proc.time()[['elapsed']]-t_start,preprocessing_seconds=t_mixstart-t_start,mixscale_signature_seconds=t_mixend-t_mixstart,checkpoint_seconds=t0-t_mixend,regression_seconds=proc.time()[['elapsed']]-t0),file.path(out_dir,'RUN_COMPLETE.json'),auto_unbox=TRUE,pretty=TRUE)
