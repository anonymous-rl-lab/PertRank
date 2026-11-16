"""Synthetic-only numerical and exclusion tests for boundary learning."""
from pathlib import Path
import argparse,json,tempfile
from fractions import Fraction
import numpy as np
from boundary_components import (torch,tensor,BoundaryNormalizer,boundary_context,
    make_boundary_model,build_pair_labels,pair_loss)
from base_training import GateNormalizer
from common import membership
from gate_model import ReliabilityGate,list_loss


def fixture(q=6,n=13,seed=902):
    rng=np.random.default_rng(seed);raw=rng.normal(size=(q,n))
    own=np.zeros((q,n),bool);own[np.arange(q),np.arange(q)%n]=True
    wm=rng.uniform(.02,.6,size=(q,n));w0=rng.uniform(.02,.6,size=(q,n))
    energy=rng.uniform(.01,2,size=(q,n));X=np.stack((raw,raw*wm,raw*w0,rng.uniform(.01,.3,(q,n)),wm,w0,np.full((q,n),20),energy,.8*energy,np.full((q,n),.2),rng.uniform(0,.5,(q,n)),np.ones((q,n)),rng.uniform(.1,.5,(q,n))),axis=2)
    return dict(X=X,raw=raw,base=raw*wm,euclidean=raw*w0,ownmask=own,feature_names=np.array([f'f{i}' for i in range(13)]),reference=rng.normal(size=(q,n)))


def manual_member(row,mask,k,sign):
    if k==0:return [Fraction(0) for _ in row]
    v=[sign*x for x,m in zip(row,mask) if not m];cut=sorted(v,reverse=True)[k-1]
    above=sum(x>cut for x in v);equal=sum(x==cut for x in v)
    return [Fraction(0) if m else Fraction(1) if sign*x>cut else Fraction(k-above,equal) if sign*x==cut else Fraction(0) for x,m in zip(row,mask)]


def run():
    checks=[]
    def passed(name,**extra):checks.append(dict(name=name,passed=True,**extra))
    v=fixture();k=3;norm=BoundaryNormalizer(v,k);old=GateNormalizer(v)
    t=norm.tensors(v,'boundary_pair',k);t0=old.tensors(v,'neural_no_extra_geometry')
    assert t['features'].shape==v['base'].shape+(14,)
    assert torch.equal(t['features'][:,:,:11],t0['features'])
    assert torch.equal(t['features'][:,:,8:11],torch.zeros_like(t['features'][:,:,8:11]))
    passed('first_eleven_columns_exact_prior_normalizer_and_geometry_predictors_zero')
    m=make_boundary_model(123)
    assert np.array_equal(m(t['features'],t['raw'],t['anchor']).detach().numpy(),v['base'])
    passed('extended_gate_exact_original_initialization')
    b=boundary_context(v['base'],v['ownmask'],k,norm.baseline_scale)
    flipped=boundary_context(-v['base'],v['ownmask'],k,norm.baseline_scale)
    assert np.array_equal(b,flipped)
    passed('boundary_features_exact_global_sign_flip_invariance')
    perm=np.random.default_rng(21).permutation(v['base'].shape[1]);inv=np.argsort(perm)
    bp=boundary_context(v['base'][:,perm],v['ownmask'][:,perm],k,norm.baseline_scale)[:,inv]
    assert np.array_equal(b,bp)
    passed('boundary_features_exact_candidate_permutation_equivariance')
    tied=np.array([[10.,8.,8.,4.,1.,-1.,-4.,-8.,-8.,-10.,999.]])
    mask=np.zeros_like(tied,bool);mask[0,-1]=True
    bt=boundary_context(tied,mask,3,np.array([2.]))
    assert np.isclose(bt[0,0,0],2) and np.isclose(bt[0,1,0],1)
    assert np.isclose(bt[0,1,1],1/3) and np.isclose(bt[0,2,1],1/3)
    assert np.all(bt[0,-1]==0)
    passed('manual_cutoff_midpoint_and_tie_midrank_check')
    poisoned={x:np.copy(a) for x,a in v.items()}
    poisoned['raw'][v['ownmask']]=1e6;poisoned['base'][v['ownmask']]=-1e6
    poisoned['X'][v['ownmask']]=1e6
    npn=BoundaryNormalizer(poisoned,k)
    for key in ('raw_scale','baseline_scale','se_floor','feature_mean','feature_scale','boundary_mean','boundary_scale'):
        assert np.array_equal(getattr(norm,key),getattr(npn,key)),key
    assert np.array_equal(norm.tensors(v,'boundary_pair',k)['features'].numpy()[~v['ownmask']],npn.tensors(poisoned,'boundary_pair',k)['features'].numpy()[~v['ownmask']])
    passed('own_target_poisoning_cannot_change_fitted_moments_or_eligible_context')
    poisoned['reference']*=1e8
    nr=BoundaryNormalizer(poisoned,k)
    for key in ('raw_scale','baseline_scale','feature_mean','feature_scale','boundary_mean','boundary_scale'):
        assert np.array_equal(getattr(npn,key),getattr(nr,key))
    passed('normalizer_never_reads_reference')
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'norm.npz';norm.save(p);restored=BoundaryNormalizer.from_file(p)
        for key,a in t.items():assert torch.equal(a,restored.tensors(v,'boundary_context',k)[key])
    passed('normalization_roundtrip_exact')
    oldmodel=ReliabilityGate(11,'neural',8,123);newmodel=make_boundary_model(123)
    z=torch.cat((t0['features'],torch.zeros(v['base'].shape+(3,))),dim=2)
    support=tensor(membership(v['reference'],v['ownmask'],k))
    opts=[torch.optim.Adam(a.parameters(),lr=.01) for a in (oldmodel,newmodel)]
    for _ in range(2):
        for model,feat,opt in zip((oldmodel,newmodel),(t0['features'],z),opts):
            opt.zero_grad();score=model(feat,t0['raw'],t0['anchor'])
            loss=list_loss(score/t0['scale'],t0['mask'],support[0],support[1],k,.5)
            loss.backward();opt.step()
    a=oldmodel(t0['features'],t0['raw'],t0['anchor']);c=newmodel(z,t0['raw'],t0['anchor'])
    difference=float(abs(a-c).max().detach());assert difference<1e-12
    passed('zero_boundary_columns_match_old_path_after_two_updates',max_difference=difference)
    b1=make_boundary_model(123);b2=make_boundary_model(123)
    assert all(torch.equal(x,y) for x,y in zip(b1.parameters(),b2.parameters()))
    assert not torch.equal(b1.h[0].weight[:,11:],make_boundary_model(124).h[0].weight[:,11:])
    passed('matched_arms_identical_seed_and_extra_columns_seeded')
    labels=build_pair_labels(v,k,support)
    assert not np.any(v['ownmask'][labels['q'],labels['i']])
    assert not np.any(v['ownmask'][labels['q'],labels['j']])
    assert torch.all(labels['i']!=labels['j'])
    assert labels['denominator']==2*len(v['base'])*4
    passed('pair_labels_self_excluded_and_fixed_denominator',stats=labels['stats'])
    sp=support.numpy().copy();sp[:,v['ownmask']]=float('nan')
    lp=build_pair_labels(v,k,sp)
    for key in ('q','i','j','direction','label','weight'):assert torch.equal(lp[key],labels[key])
    passed('own_reference_poison_has_no_effect')
    const=np.broadcast_to((k/(~v['ownmask']).sum(1))[None,:,None],support.shape)
    weak=.1*support.numpy()+.9*const
    weak=np.where(v['ownmask'][None],0,weak)
    lw=build_pair_labels(v,k,weak)
    score=t['base']/t['scale'];strongloss=pair_loss(score,labels);weakloss=pair_loss(score,lw)
    assert np.isclose(float(weakloss),.1*float(strongloss),rtol=1e-12,atol=1e-14)
    passed('low_confidence_scales_loss_without_renormalizing_it_away',ratio=float(weakloss/strongloss))
    zero=np.where(v['ownmask'][None],0,const)
    lz=build_pair_labels(v,k,zero);zs=score.clone().requires_grad_(True);loss=pair_loss(zs,lz)
    loss.backward();assert float(loss.detach())==0 and torch.equal(zs.grad,torch.zeros_like(zs))
    passed('equal_reference_support_zero_loss_and_gradient')
    ls=score.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x:pair_loss(x,labels),(ls,),eps=1e-6,atol=1e-6,rtol=1e-4)
    passed('pair_loss_finite_difference_gradient')
    ls=score.clone().requires_grad_(True);pair_loss(ls,labels).backward()
    assert torch.equal(ls.grad[t['mask']==False],torch.zeros_like(ls.grad[t['mask']==False]))
    passed('own_score_gradient_exact_zero')
    mt=make_boundary_model(123);pl=pair_loss(mt(t['features'],t['raw'],t['anchor'])/t['scale'],labels);pl.backward()
    gradient=float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in mt.h[-1].parameters()])))
    assert gradient>1e-8
    passed('pair_objective_has_nonzero_output_head_gradient_at_original_anchor',gradient_norm=gradient)
    vv=dict(v);vv['base']=-v['base'];lf=build_pair_labels(vv,k,support.flip(0))
    assert np.isclose(float(pair_loss(-score,lf)),float(strongloss),rtol=1e-14,atol=1e-15)
    passed('pair_loss_global_sign_flip_with_reference_directions_swapped')
    vp=dict(v);vp['base']=v['base'][:,perm];vp['ownmask']=v['ownmask'][:,perm]
    lp=build_pair_labels(vp,k,support.numpy()[:,:,perm])
    assert np.isclose(float(pair_loss(score[:,perm],lp)),float(strongloss),rtol=1e-14,atol=1e-15)
    passed('pair_loss_candidate_permutation_invariance')
    # A fully independent rational construction on exact tied scores.
    row=tied[0];own=mask[0];ref=np.array([0,2,2,-1,3,-2,4,0,0,-3,500.])
    spi=np.array([[float(x) for x in manual_member(ref,own,3,sg)] for sg in (1.,-1.)])[:,None,:]
    lab=build_pair_labels({'base':tied,'ownmask':mask},3,spi)
    expected=0.
    for d,sg in enumerate((1.,-1.)):
        mm=[manual_member(row,own,kk,sg) for kk in (1,3,5)]
        ain=[b-a for a,b in zip(mm[0],mm[1])];bout=[b-a for a,b in zip(mm[1],mm[2])]
        for i in range(len(row)):
            for j in range(len(row)):
                if i==j:continue
                delta=spi[d,0,i]-spi[d,0,j]
                weight=float(ain[i]*bout[j])*abs(delta)
                margin=np.sign(delta)*sg*(row[i]-row[j])/.5
                expected+=weight*np.logaddexp(0.,-margin)/8
    actual=float(pair_loss(tensor(tied),lab));assert np.isclose(actual,expected,rtol=1e-14,atol=1e-15)
    passed('independent_fractional_tie_dense_pair_loss',actual=actual,expected=expected)
    assert np.isclose(labels['stats']['total_confidence_weight'],sum(labels['stats'][k] for k in ('correct_confidence_weight','wrong_confidence_weight','tied_confidence_weight')))
    passed('confidence_mass_correct_wrong_tied_accounting')
    return dict(all_passed=True,synthetic_only=True,real_reference_access=False,torch_version=str(torch.__version__),checks=checks)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).with_name('BOUNDARY_CHECKS.json'));a=p.parse_args()
    report=run();a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
