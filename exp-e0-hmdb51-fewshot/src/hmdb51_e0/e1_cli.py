"""Explicit E1 stages; no fitting or text encoding occurs implicitly."""
import argparse
import json
from pathlib import Path
import sys

from .config import ROOT,load_paths
from .e1_features import load_e1_config,validate_text_alignment
from .pipeline import official_clip_condition
from .provenance import new_directory,read_json
from .round2_io import safe_output


def require_official_e1(path):
    from .e1_experiment import OFFICIAL
    index=read_json(path)
    if index.get('feature_condition')!=OFFICIAL or not official_clip_condition(index.get('identity',{})):
        raise ValueError('production E1 requires official image AND native text features')
    validate_text_alignment(index['identity']['encoder'],index['identity']['text_encoder'],index['text_frozen_verification'])


def build_parser():
    parser=argparse.ArgumentParser(description='E1 C0/C1 and matched R0/R1 controls; final test sealed')
    commands=parser.add_subparsers(dest='command',required=True)
    for name,help_text in (('check','Read-only reference/cache/checkpoint check; no text encoding or fitting'),
                           ('prepare','Encode fixed concepts and derive E1 cache; no image encoding or fitting'),
                           ('run','Explicitly fit all E1 development conditions'),
                           ('summarize','Validate all E1 runs and recompute paired metrics')):
        sub=commands.add_parser(name,help=help_text,description=help_text)
        sub.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
        if name in ('check','prepare'):
            sub.add_argument('--reference-batch',type=Path,required=True)
            sub.add_argument('--config',type=Path,default=ROOT/'configs/e1.baseline.yaml')
            sub.add_argument('--concepts',type=Path,default=ROOT/'configs/e1.concepts.v1.yaml')
            sub.add_argument('--feature-root',type=Path,default=ROOT/'artifacts/round2_features')
            if name=='prepare':
                sub.add_argument('--output',type=Path,default=ROOT/'artifacts/e1_concepts')
                sub.add_argument('--device',choices=('cpu','cuda'),default='cuda')
        elif name=='run':
            sub.add_argument('--reference-batch',type=Path,required=True)
            sub.add_argument('--feature-index',type=Path,required=True)
            sub.add_argument('--stage',choices=('development',),default='development')
        else:
            sub.add_argument('--batch',type=Path,required=True)
            sub.add_argument('--output',type=Path,help='Report parent; creates a new child directory')
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        paths=load_paths(args.paths)
        if args.command in ('check','prepare','run'):
            from .cli import require_official_features
            require_official_features(args.reference_batch/'feature_index_snapshot.json')
        if args.command in ('check','prepare'):
            from .e1_experiment import check_e1,prepare_e1_features
            config=load_e1_config(args.config)
            if args.command=='check':
                print(json.dumps(check_e1(args.reference_batch,config,args.concepts,paths,args.feature_root),ensure_ascii=False,indent=2)); return 0
            result=prepare_e1_features(args.reference_batch,config,args.concepts,paths,args.feature_root,args.output,device=args.device)
        elif args.command=='run':
            from .e1_experiment import run_e1
            require_official_e1(args.feature_index)
            result=run_e1(args.reference_batch,args.feature_index,paths,stage=args.stage)
        else:
            from .e1_reporting import summarize_e1,validate_e1_runs
            validate_e1_runs(args.batch)
            parent=safe_output(args.output or paths['report_root'],paths,protected=(args.batch,))
            folder=new_directory(parent,'e1-concept-report')
            summarize_e1(args.batch,folder); result=folder/'e1-results.md'
        print(result); return 0
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr); return 2
