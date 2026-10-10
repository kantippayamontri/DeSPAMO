"""Matched plain-MLP fusion/FLAN-LoRA model and standalone update kernel."""

import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from despamo.losses.vt_align import VTAlignLoss
from despamo.models.aligned_fusion import enable_aligned_fusion
from despamo.models.flan_t5 import FlanT5Backbone
from despamo.models.visual_adapter import SpaMoVisualAdapter
from despamo.training.baseline_module import SpaMoBaselineModule
from tools.benchmark.compact_checkpoint import frozen_fingerprint

FLAN_REVISION="7d6315df2c2fb742f0f5b556879d730926ca9001"


def schedule(batch_size: int) -> dict:
    if batch_size not in (4,8):
        raise ValueError("profiled translation batches4/8 only")
    per=7096//batch_size
    return dict(batch_size=batch_size,steps_per_epoch=per,total_steps=40*per,epochs=40,
                dev_epochs=list(range(2,41,2)),vt_warmup_steps=2000,lr_warmup_steps=4*per)


def lr_factor(k: int,total_steps: int) -> float:
    if type(k)is not int or not 1<=k<=total_steps:
        raise ValueError("one-based update outside declared horizon")
    warm=total_steps//10
    if k<=warm:
        return k/warm
    return .5*(1+math.cos(math.pi*(k-warm)/(total_steps-warm)))


def plain_adapter(base,seed: int):
    with torch.random.fork_rng():
        torch.manual_seed(100+seed)
        base.spatial_projector=nn.Sequential(nn.Linear(2048,1024),nn.GELU(),nn.Linear(1024,768))
    return base


def build_model(seed: int,arm: str,cache: str):
    if seed not in (0,1) or arm not in ("reference","aligned_fixed","aligned_learned"):
        raise ValueError("approved model/seed required")
    torch.manual_seed(seed)
    language=FlanT5Backbone.from_pretrained("google/flan-t5-xl",cache,64,"lora",16,32,.1,
                                           revision=FLAN_REVISION)
    visual=plain_adapter(SpaMoVisualAdapter(2048,1024,768,2048),seed)
    visual=enable_aligned_fusion(visual,arm,seed)
    model=SpaMoBaselineModule(visual,language,VTAlignLoss(),"Translate the given sentence into {}.",
                              False,0,"masked_mean",1.,2000,3e-4,.01,seed)
    base_hash=frozen_fingerprint(model)
    return model,base_hash


def optimizer_for(model):
    adapter=[]
    language=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            (language if name.startswith("language_model.") else adapter).append(p)
    return torch.optim.AdamW([dict(name="adapter",params=adapter,lr=3e-4),
                             dict(name="flan",params=language,lr=5e-5)],
                             betas=(.9,.98),eps=1e-8,weight_decay=.01)


def cuda_batch(batch):
    return replace(batch,spatial=batch.spatial.to("cuda"),motion=batch.motion.to("cuda"),
                    spatial_mask=batch.spatial_mask.to("cuda"),motion_mask=batch.motion_mask.to("cuda"))


def update(model,optimizer,batch,k: int,recipe: dict,*,force_joint: bool=False) -> dict:
    factor=lr_factor(k,recipe["total_steps"])
    for group in optimizer.param_groups:
        group["lr"]=(5e-5 if group["name"]=="flan" else 3e-4)*factor
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(batch.spatial.device.type,dtype=torch.bfloat16):
        visual,valid=model.visual_adapter(batch.spatial,batch.spatial_mask,
                                          batch.motion,batch.motion_mask)
        text,mask=model.language_model.target_embeddings(batch.texts)
        vt=model.vt_align(visual.float(),valid,text.float(),mask,"masked_mean")
        ce=None
        if force_joint or k>recipe["vt_warmup_steps"]:
            ce=model.language_model.translation_loss(visual,valid,
                [model.prompt_template.format("German") for _ in batch.clip_ids],batch.texts)
        loss=vt if ce is None else ce+vt
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite translation loss")
    loss.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
        raise FloatingPointError("non-finite translation gradients")
    norm=clip_grad_norm_(params,1.)
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite translation norm")
    optimizer.step()
    return dict(loss=float(loss.detach()),vt_loss=float(vt.detach()),
                translation_loss=None if ce is None else float(ce.detach()),grad_norm=float(norm),
                adapter_lr=3e-4*factor,flan_lr=5e-5*factor,step=k)
