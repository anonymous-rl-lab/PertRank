"""Input-only list context and calibration-only local boundary ranking loss.

List context uses the available predictor pool, excluding query-self targets.
Standardization moments use only the fit pool. Empirical reference-membership
stability supplies loss labels, never features or latent-truth probabilities.
"""
from __future__ import annotations
import math
import numpy as np
from base_training import GateNormalizer, tensor, torch, load, FEATURES
from common import membership
from gate_model import ReliabilityGate
BOUNDARY_FEATURES=('tail_score_depth','tail_rank_depth','cutoff_crowding')
ALL_FEATURES=tuple(FEATURES)+BOUNDARY_FEATURES


def boundary_context(base,own,k,baseline_scale):
    """Three sign-flip-equivariant columns from eligible inputs only."""
    base=np.asarray(base,dtype=float);own=np.asarray(own,dtype=bool)
    scale=np.asarray(baseline_scale,dtype=float).reshape(-1)
    if base.ndim!=2 or own.shape!=base.shape or scale.shape!=(len(base),):
        raise ValueError('Invalid predictor/mask/scale dimensions')
    if not isinstance(k,(int,np.integer)) or k<1 or np.any(scale<=0) or not np.isfinite(scale).all():
        raise ValueError('Positive K and finite positive scales required')
    out=np.zeros(base.shape+(3,),dtype=float)
    for q in range(len(base)):
        ix=np.flatnonzero(~own[q]);v=base[q,ix]
        if len(v)<=k or not np.isfinite(v).all():raise ValueError('Need at least K+1 finite eligible candidates')
        order=np.sort(v)
        upper=.5*order[-k]+.5*order[-k-1]
        lower=.5*order[k-1]+.5*order[k]
        du=(v-upper)/scale[q];dd=(lower-v)/scale[q]
        equal=(v[:,None]==v[None,:]).sum(1)
        up=(v[None,:]>v[:,None]).sum(1)+(equal+1)/2
        down=(v[None,:]<v[:,None]).sum(1)+(equal+1)/2
        pu=float((np.abs(v-upper)<=.5*scale[q]).mean())
        pd=float((np.abs(v-lower)<=.5*scale[q]).mean())
        crowd=np.where(du>dd,pu,np.where(dd>du,pd,.5*(pu+pd)))
        out[q,ix,0]=np.maximum(du,dd)
        out[q,ix,1]=((k+.5)-np.minimum(up,down))/k
        out[q,ix,2]=crowd
    assert np.isfinite(out).all()
    return out


class BoundaryNormalizer(GateNormalizer):
    """Old no-extra-geometry normalization plus three input-only columns."""
    n_features=14
    def __init__(self,v,k):
        super().__init__(v)
        self.fit_k=int(k)
        b=boundary_context(v['base'],v['ownmask'],k,self.baseline_scale)
        valid=~v['ownmask']
        self.boundary_mean=b[valid].mean(0)
        self.boundary_scale=np.maximum(b[valid].std(0),1e-6)
    def tensors(self,v,arm,k):
        result=super().tensors(v,'neural_no_extra_geometry')
        b=boundary_context(v['base'],v['ownmask'],k,self.baseline_scale)
        result['features']=torch.cat((result['features'],tensor((b-self.boundary_mean)/self.boundary_scale)),dim=2)
        return result
    def save(self,p):
        np.savez_compressed(p,source_feature_names=self.source_feature_names,
            feature_names=np.array(ALL_FEATURES),raw_scale=self.raw_scale,
            baseline_scale=self.baseline_scale,se_floor=self.se_floor,
            feature_mean=self.feature_mean,feature_scale=self.feature_scale,
            boundary_mean=self.boundary_mean,boundary_scale=self.boundary_scale,
            fit_k=np.array(self.fit_k))
    @classmethod
    def from_file(cls,p):
        obj=cls.__new__(cls)
        for key,value in load(p).items():setattr(obj,key,value)
        obj.fit_k=int(obj.fit_k)
        return obj


def make_boundary_model(seed):
    """Preserve the old 11-column hidden initialization; add seeded columns."""
    old=ReliabilityGate(11,'neural',8,seed)
    new=ReliabilityGate(14,'neural',8,seed)
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(int(seed)+100003)
        new.h[0].weight[:,:11].copy_(old.h[0].weight)
        new.h[0].bias.copy_(old.h[0].bias)
        new.h[0].weight[:,11:].uniform_(-1/math.sqrt(11),1/math.sqrt(11))
        new.h[-1].weight.copy_(old.h[-1].weight)
        new.h[-1].bias.copy_(old.h[-1].bias)
    return new


def _members(base,own,k):
    return np.zeros((2,)+base.shape,dtype=float) if k==0 else membership(base,own,k)


def build_pair_labels(v,k,support):
    """Sparse directed local-band edges from current fit-pool references.

    In/out bands each have mass two. Their product is a fractional boundary
    proxy, not an exact probability of a simultaneous exchange. Correct and
    incorrect comparisons both appear. Equal empirical support gets zero
    weight. Always divide by2*Q*4, retaining low-confidence small differences.
    """
    base=np.asarray(v['base'],float);own=np.asarray(v['ownmask'],bool)
    if not isinstance(k,(int,np.integer)) or k<2 or np.any((~own).sum(1)<k+2):
        raise ValueError('Need K>=2 and at least K+2 eligible candidates')
    if isinstance(support,torch.Tensor):support=support.detach().cpu().numpy()
    pi=np.asarray(support,float)
    if pi.shape!=(2,)+base.shape:raise ValueError('Support must be[2,Q,T]')
    pi=np.where(own[None],0.,pi)
    if not np.isfinite(pi).all() or np.any(pi < -1e-12) or np.any(pi>1+1e-12):
        raise ValueError('Eligible support must be finite in[0,1]')
    if np.max(np.abs(pi.sum(2)-k))>1e-8:
        raise ValueError('Reference support must sum to current K per query/direction')
    mid=_members(base,own,k)
    inside=mid-_members(base,own,k-2)
    outside=_members(base,own,k+2)-mid
    assert np.min(inside)>-1e-12 and np.min(outside)>-1e-12
    assert np.max(abs(inside.sum(2)-2))<1e-10
    assert np.max(abs(outside.sum(2)-2))<1e-10
    arrays={key:[] for key in ('q','i','j','direction','label','weight')}
    structural_count=0;band_mass=0.;correct=wrong=tied=0.
    for d,direction in enumerate((1.,-1.)):
        for q in range(len(base)):
            band=inside[d,q,:,None]*outside[d,q,None,:]
            np.fill_diagonal(band,0.)
            ii,jj=np.nonzero(band>0)
            structural_count+=len(ii);band_mass+=float(band.sum())
            diff=pi[d,q,ii]-pi[d,q,jj]
            take=diff!=0;ii=ii[take];jj=jj[take];diff=diff[take]
            weight=band[ii,jj]*np.abs(diff);label=np.sign(diff)
            alignment=direction*(base[q,ii]-base[q,jj])*label
            correct+=float(weight[alignment>0].sum());wrong+=float(weight[alignment<0].sum());tied+=float(weight[alignment==0].sum())
            for key,value in [('q',np.full(len(ii),q)),('i',ii),('j',jj),('direction',np.full(len(ii),direction)),('label',label),('weight',weight)]:
                arrays[key].append(value)
    labels={}
    for key,parts in arrays.items():
        a=np.concatenate(parts) if parts else np.empty(0)
        labels[key]=torch.as_tensor(a,dtype=torch.int64 if key in ('q','i','j') else torch.float64)
    labels['denominator']=float(2*len(base)*4)
    labels['stats']=dict(queries=len(base),k=int(k),edge_count=len(labels['q']),
        structural_edge_count=structural_count,structural_band_mass=band_mass,
        total_confidence_weight=float(labels['weight'].sum()),correct_confidence_weight=correct,
        wrong_confidence_weight=wrong,tied_confidence_weight=tied,denominator=labels['denominator'],
        reference_identity='Empirical donor-bootstrap TopK membership stability, not latent truth')
    return labels


def pair_loss(score_normalized,labels,tau=.5):
    """Weighted logistic comparisons on frozen-scaled physical scores."""
    if tau<=0:raise ValueError('Positive temperature required')
    if len(labels['q'])==0:return score_normalized.sum()*0.
    q,i,j=labels['q'],labels['i'],labels['j']
    margin=labels['label']*labels['direction']*(score_normalized[q,i]-score_normalized[q,j])/tau
    return (labels['weight']*torch.nn.functional.softplus(-margin)).sum()/labels['denominator']
