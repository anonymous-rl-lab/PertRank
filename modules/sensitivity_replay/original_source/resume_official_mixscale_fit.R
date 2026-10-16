#!/usr/bin/env Rscript
# Official fits from the identical full-data fitted signature object; sharding changes no scientific input.
a=commandArgs(trailingOnly=TRUE)
if(length(a)!=5) stop('usage: resume_official_mixscale_fit.R <scored.rds> <input_dir> <out_dir> <start> <end>')
suppressPackageStartupMessages({library(Seurat);library(Mixscale);library(jsonlite)})
obj=readRDS(a[[1]]);invisible(gc());ip=a[[2]];op=a[[3]];lo=as.integer(a[[4]]);hi=as.integer(a[[5]]);dir.create(op,recursive=TRUE,showWarnings=FALSE)
qs=read.csv(file.path(ip,'queries.csv'),stringsAsFactors=FALSE);fmap=read.csv(file.path(dirname(a[[1]]),'feature_name_map.csv'),stringsAsFactors=FALSE)
q=fmap$actual_feature_id[match(qs$feature_id,fmap$source_feature_id)];stopifnot(!anyNA(q),!anyDuplicated(q))
targets=read.csv(file.path(ip,'targets.csv'),stringsAsFactors=FALSE);stopifnot(nrow(targets)==1200,lo>=1,hi<=1200,lo<=hi)
st=list()
for(i in lo:hi){
 set.seed(20261005+i)
 tg=targets$target_group[[i]];tt=proc.time()[['elapsed']]
 sf=file.path(op,sprintf('status_%04d.json',i));cf=file.path(op,sprintf('target_%04d.csv',i))
 if(file.exists(sf)&&file.exists(cf)){old=read_json(sf,simplifyVector=TRUE);oldcsv=read.csv(cf,check.names=FALSE);stopifnot(old$index==i,old$target_group==tg,nrow(oldcsv)==old$n_query_returned,all(is.finite(oldcsv$beta_weight)),!anyDuplicated(oldcsv$gene_ID));cat(i,'existing verified checkpoint, skip\n');next}
 rr=Run_wmvRegDE(object=obj,assay='RNA',slot='counts',labels='target_group',nt.class.name='NT',PRTB_list=tg,logfc.threshold=0,min.pct=.1,min.cells.group=10,total_ct_labels='nCount_RNA',split.by=NULL,verbose=FALSE,full.results=FALSE)
 stopifnot(length(rr)==1,tg %in% names(rr));r=rr[[tg]];stopifnot(all(c('gene_ID','beta_weight','DE_method') %in% names(r)),!anyDuplicated(r$gene_ID))
 keep=r$gene_ID %in% q;if(any(!is.finite(r$beta_weight[keep]))) stop('nonfinite native coefficient')
 write.csv(r[keep,,drop=FALSE],file.path(op,sprintf('target_%04d.csv',i)),row.names=FALSE)
 st[[length(st)+1]]=list(index=i,target_group=tg,n_query_returned=sum(keep),n_all_returned=nrow(r),fit_methods=unique(r$DE_method),seconds=proc.time()[['elapsed']]-tt)
 write_json(st[[length(st)]],file.path(op,sprintf('status_%04d.json',i)),auto_unbox=TRUE,pretty=TRUE)
 write_json(st,file.path(op,sprintf('target_status_%04d_%04d.json',lo,hi)),auto_unbox=TRUE,pretty=TRUE)
 invisible(gc())
 cat(i,tg,'complete\n');flush.console()
}
