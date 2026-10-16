#!/usr/bin/env Rscript
# Same pinned native algorithm on exact stored global-context target/NTC blocks.
a=commandArgs(trailingOnly=TRUE);stopifnot(length(a)==8)
helper=a[[1]];compact_dir=a[[2]];in_dir=a[[3]];eq_dir=a[[4]];out=a[[5]];lo=as.integer(a[[6]]);hi=as.integer(a[[7]]);workers=as.integer(a[[8]])
suppressPackageStartupMessages({library(Seurat);library(Mixscale);library(Matrix);library(jsonlite)})
source(helper);stopifnot(workers %in% c(1L,2L,3L,4L),lo>=1L,hi<=1200L,lo<=hi)
eq=read_json(file.path(eq_dir,'COMPACT_EQUIVALENCE_PASS.json'),simplifyVector=TRUE);stopifnot(isTRUE(eq$passed))
release_gate_path=file.path(dirname(eq_dir),'de_slot_release_gate','DE_SLOT_RELEASE_EQUIVALENCE.json')
release_gate=read_json(release_gate_path,simplifyVector=TRUE);stopifnot(isTRUE(release_gate$passed),nrow(release_gate$targets)==4)
shared=readRDS(file.path(compact_dir,'shared_ntc.rds'));targets=read.csv(file.path(in_dir,'targets.csv'),stringsAsFactors=FALSE);stopifnot(nrow(targets)==1200)
anchor_tool=readRDS(file.path(eq_dir,'anchor_score_context.rds'));stopifnot(length(anchor_tool)==1)
fm=read.csv(file.path(dirname(shared$source$path),'feature_name_map.csv'),stringsAsFactors=FALSE);qs=read.csv(file.path(in_dir,'queries.csv'),stringsAsFactors=FALSE);query_ids=fm$actual_feature_id[match(qs$feature_id,fm$source_feature_id)];stopifnot(!anyNA(query_ids),!anyDuplicated(query_ids))
dir.create(out,recursive=TRUE,showWarnings=FALSE);write.csv(fm,file.path(out,'feature_name_map.csv'),row.names=FALSE)
input_manifest=read.csv(file.path(compact_dir,'compact_manifest.csv'),stringsAsFactors=FALSE);stopifnot(nrow(input_manifest)==1200)
t0=proc.time()[['elapsed']]
valid_existing=function(i){sf=file.path(out,sprintf('status_%04d.json',i));cf=file.path(out,sprintf('target_%04d.csv',i));if(!file.exists(sf)||!file.exists(cf))return(FALSE);s=read_json(sf,simplifyVector=TRUE);r=read.csv(cf,check.names=FALSE);stopifnot(s$index==i,s$target_group==targets$target_group[[i]],nrow(r)==s$n_query_returned,!anyDuplicated(r$gene_ID),all(is.finite(r$beta_weight)),identical(s$execution_adapter,'verified target-local exact representation'));TRUE}
fit_one=function(i){
 set.seed(20261005+i);tg=targets$target_group[[i]];tt=proc.time()[['elapsed']]
 obj=load_compact(shared,file.path(compact_dir,sprintf('target_%04d.rds',i)))
 stopifnot(identical(as.character(unique(obj$target_group)),c('NT',tg)),ncol(obj)==2020,nrow(obj[['RNA']])==18130)
 invisible(gc()) # execution only: release construction temporaries
 obj=RunMixscale(object=obj,assay='PRTB',slot='scale.data',labels='target_group',nt.class.name='NT',min.de.genes=5,min.cells=5,logfc.threshold=.2,de.assay='RNA',max.de.genes=100,new.class.name='mixscale_score',fine.mode=FALSE,harmonize=FALSE,pval.cutoff=.05,seed=10282021,verbose=FALSE,split.by=NULL)
 native_scores=Tool(obj,slot='RunMixscale');scored=tg %in% names(native_scores);n_sig=if(scored)ncol(native_scores[[tg]][[1]])-2 else 0L
 # Only metadata context for the native unscored-target branch; it is excluded
 # from the target list and never supplies weights to this target.
 if(!names(anchor_tool)[[1]] %in% names(native_scores))native_scores=c(anchor_tool,native_scores)
 obj@tools$RunMixscale=native_scores
 # Exact native-DE execution gate permits releasing only these unread slots.
 stopifnot(DefaultAssay(obj)=='RNA');obj@assays$PRTB=NULL;obj@assays$RNA@scale.data=matrix(numeric(),nrow=0L,ncol=0L);methods::validObject(obj)
 invisible(gc());ts=proc.time()[['elapsed']]
 set.seed(20261005+i)
 rr=Run_wmvRegDE(object=obj,assay='RNA',slot='counts',labels='target_group',nt.class.name='NT',PRTB_list=tg,logfc.threshold=0,min.pct=.1,min.cells.group=10,total_ct_labels='nCount_RNA',split.by=NULL,verbose=FALSE,full.results=FALSE)
 stopifnot(length(rr)==1,tg %in% names(rr));r=rr[[tg]];stopifnot(all(c('gene_ID','beta_weight','DE_method')%in%names(r)),!anyDuplicated(r$gene_ID));keep=r$gene_ID %in% query_ids;stopifnot(all(is.finite(r$beta_weight[keep])))
 cf=file.path(out,sprintf('target_%04d.csv',i));write.csv(r[keep,,drop=FALSE],paste0(cf,'.tmp'),row.names=FALSE);stopifnot(file.rename(paste0(cf,'.tmp'),cf))
 mem=readLines('/proc/self/status');st=list(index=i,target_group=tg,n_query_returned=sum(keep),n_all_returned=nrow(r),fit_methods=unique(r$DE_method),native_score_present=scored,n_signature_genes=n_sig,signature_seconds=ts-tt,regression_seconds=proc.time()[['elapsed']]-ts,seconds=proc.time()[['elapsed']]-tt,execution_adapter='verified target-local exact representation',de_execution_adapter='unused PRTB and RNA scale.data released after four-target exact native DE gate',process_memory=mem[grepl('^Vm(RSS|HWM):',mem)])
 sf=file.path(out,sprintf('status_%04d.json',i));write_json(st,paste0(sf,'.tmp'),pretty=TRUE,auto_unbox=TRUE);stopifnot(file.rename(paste0(sf,'.tmp'),sf));cat('COMPLETE',i,1200,tg,'weighted',scored,'seconds',st$seconds,'\n');flush.console();st
}
todo=(lo:hi)[!vapply(lo:hi,valid_existing,logical(1))];cat('TODO',length(todo),'workers',workers,'range',lo,hi,'\n');flush.console()
run_safe=function(i)tryCatch(fit_one(i),error=function(e){write_json(list(index=i,target_group=targets$target_group[[i]],error=conditionMessage(e)),file.path(out,sprintf('error_%04d.json',i)),pretty=TRUE,auto_unbox=TRUE);stop(e)})
if(length(todo)>0){if(workers==1L)st=lapply(todo,run_safe)else st=parallel::mclapply(todo,run_safe,mc.cores=workers,mc.preschedule=FALSE,mc.set.seed=FALSE);if(any(vapply(st,inherits,logical(1),'try-error')))stop('At least one native fit failed; partial predictions must not be evaluated.')}
write_json(list(range=c(lo,hi),workers=workers,completed_targets=length(todo),seconds=proc.time()[['elapsed']]-t0),file.path(out,sprintf('batch_%04d_%04d.json',lo,hi)),pretty=TRUE,auto_unbox=TRUE)
allgood=vapply(seq_len(1200),valid_existing,logical(1));if(all(allgood)){
 st=lapply(seq_len(1200),function(i)read_json(file.path(out,sprintf('status_%04d.json',i)),simplifyVector=TRUE));write_json(st,file.path(out,'target_status.json'),pretty=TRUE,auto_unbox=TRUE)
 write_json(list(mode='full',n_targets=1200,source_commit='3d3a2cd12e62382df01c056536d967534b2de3a9',execution_adapter='verified target-local exact representation of original global preprocessing',workers_final_batch=workers,n_native_weighted=sum(vapply(st,`[[`,logical(1),'native_score_present')),regression_target_seconds_sum=sum(vapply(st,`[[`,numeric(1),'regression_seconds')),signature_target_seconds_sum=sum(vapply(st,`[[`,numeric(1),'signature_seconds')),equivalence_gate=file.path(eq_dir,'COMPACT_EQUIVALENCE_PASS.json'),de_slot_release_gate=release_gate_path),file.path(out,'RUN_COMPLETE.json'),pretty=TRUE,auto_unbox=TRUE)
}
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
