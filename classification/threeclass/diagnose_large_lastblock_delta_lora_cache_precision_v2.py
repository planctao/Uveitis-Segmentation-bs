"""Frozen diagnostic of observed cache nonfinites; preserve the original failed proof."""
import hashlib,json,os,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from experiment import ROOT,dump
from extract_dinov3_large import Images,make_model
OUT=ROOT/'runs/large_lastblock_delta_lora_cache_precision_diagnostic_v2'
DEST=ROOT/'results/large_lastblock_delta_lora_cache_precision_diagnostic_v2.json'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(protocol):
 for section in ['code_sha256','input_artifact_sha256','package_implementation_sha256']:
  for name,expected in protocol[section].items():
   p=Path(name);assert sha(p if p.is_absolute() else ROOT/p)==expected,name
@torch.inference_mode()
def main():
 assert not DEST.exists()
 torch.set_num_threads(4);protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
 frame=pd.read_csv(ROOT/'metadata/manifest.csv');ids=np.asarray(protocol['indices'],dtype=int)
 assert len(ids)==8 and frame.iloc[ids].split.eq('train').all()
 model=make_model(json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text()));capture={}
 def hook(module,args,kwargs):
  capture['source']=args[0].detach().clone();capture['rope']=kwargs['rope'].detach().clone()
 handle=model.blocks[-1].register_forward_pre_hook(hook,with_kwargs=True)
 _,*views=next(iter(DataLoader(Images(frame,ids),batch_size=8,num_workers=0)))
 rows=[]
 try:
  for size,image in zip([320,448],views):
   with torch.autocast('cuda',dtype=torch.float16):tokens=model.forward_features(image.cuda())
   source=capture['source'];half=source.half();rope=capture['rope']
   row={'size':size,'source_dtype':str(source.dtype),'shape':list(source.shape),'source_all_finite':bool(torch.isfinite(source).all()),
    'source_maximum_absolute':float(source.abs().max()),'prefix_maximum_absolute':float(source[:,:5].abs().max()),'patch_maximum_absolute':float(source[:,5:].abs().max()),
    'half_nonfinite_total':int((~torch.isfinite(half)).sum()),'half_nonfinite_prefix':int((~torch.isfinite(half[:,:5])).sum()),'half_nonfinite_patches':int((~torch.isfinite(half[:,5:])).sum()),
    'float32_all_finite':bool(torch.isfinite(source.float()).all()),'rope_all_finite':bool(torch.isfinite(rope).all()),'original_final_tokens_all_finite':bool(torch.isfinite(tokens).all())}
   rows.append(row);print(json.dumps(row),flush=True)
 finally:handle.remove()
 verify(protocol)
 overflow=all(row['source_all_finite'] and row['rope_all_finite'] and row['original_final_tokens_all_finite'] and row['float32_all_finite'] for row in rows) and any(row['half_nonfinite_total']>0 for row in rows)
 dump(DEST,{'completed':True,'created_from_frozen_diagnostic':True,'rows':rows,'float16_storage_overflow_confirmed':overflow,
  'original_failed_proof_preserved':True,'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':protocol['code_sha256'],'input_artifact_sha256':protocol['input_artifact_sha256'],
  'actual_training_images':8,'actual_full_backbone_view_evaluations':16,'classification_fits':0,'classification_scores':0,'external_raw_images_read':0,
  'scope':'Locate observed nonfinite cache conversion using original first8 training inputs; source/half/float32/rope/final-token finite and range checks. No repaired proof or classifier result.'})
if __name__=='__main__':
 dump(OUT/'state.json',{'status':'running_precision_diagnostic','pid':os.getpid(),'classification_fits':0,'classification_scores':0})
 try:
  main();dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'classification_fits':0,'classification_scores':0})
 except BaseException:
  dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'traceback':traceback.format_exc(),'classification_fits':0,'classification_scores':0});raise
