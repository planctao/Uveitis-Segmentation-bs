"""Package full-data models separately from cross-validation results."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/white_spot_20261109'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT);ap.add_argument('--runs',nargs='+',default=['tiny_s3407','tiny_s7','tiny_s3407_shift64','tiny_s7_shift64']);args=ap.parse_args();dest=args.output/'deployment';dest.mkdir(exist_ok=True)
    members=[]
    for run in args.runs:
        ck=torch.load(args.output/run/'full/best.pt',map_location='cpu',weights_only=True);assert ck['fold']==-1
        ck['source_run']=run;ck['target']='macula_center';members.append(ck)
    torch.save({'method':'tiny_ensemble','members':members,'size':384,'target':'macula_center','fit':'all 102 original image families, augmentation-family balanced; not a held-out model'},dest/'ensemble_model.pt')
    # Also expose the best-performing single-model architecture as a lightweight fallback.
    single=torch.load(args.output/'tiny_s3407/full/best.pt',map_location='cpu',weights_only=True)
    single['target']='macula_center'
    torch.save(single,dest/'model.pt')
    shutil.copyfile(dest/'model.pt',dest/'single_model.pt')
    shutil.copyfile(args.output/'posterior_classifier.joblib',dest/'posterior_classifier.joblib')
    shutil.copyfile(ROOT/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',dest/'posterior_backbone.pth')
    info={'runs':args.runs,'target':'macula_center','annotation_semantics':'User confirmed the 10-pixel fovea marker is the center of the intended macula region.','negative_class_semantics':'peripheral view, used to filter views without the annotated macula center','classifier_threshold':.5,'localizer_input_size':384,'output_coordinates':'original image pixels','marker_warning':'10-pixel circle marks the macula center; no macula extent was annotated','default_model':'single tiny_s3407 (ensemble retained as an alternative)','cv_results':'../tiny_s3407/stress_summary.json','full_fit_warning':'These models saw all supplied annotated image families. Their predictions on those images are not independent evaluation.'}
    info['files']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(dest.iterdir()) if p.is_file() and p.name!='model_card.json'}
    (dest/'model_card.json').write_text(json.dumps(info,ensure_ascii=False,indent=2));print(json.dumps(info,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
