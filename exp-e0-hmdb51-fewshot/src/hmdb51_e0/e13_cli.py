"""Explicit official-only E1.3 checks, text encoding, evaluation and reporting."""
import argparse
import json
from pathlib import Path
import sys

from .config import ROOT,load_paths
from .e13_features import load_config
from .provenance import read_json,new_directory
from .round2_io import safe_output


def build_parser():
    parser=argparse.ArgumentParser(description='E1.3 fixed class prompts; T0 once plus80 support prototype evaluations')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('check','prepare','run','summarize'):
        command=sub.add_parser(name);command.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
        if name!='summarize':command.add_argument('--e12-batch',type=Path,required=True)
        if name in ('check','prepare'):command.add_argument('--config',type=Path,default=ROOT/'configs/e13.class-text.yaml')
        if name=='prepare':
            command.add_argument('--output',type=Path,default=ROOT/'artifacts/e13_class_text')
            command.add_argument('--device',choices=('cpu','cuda'),default='cuda')
        if name=='run':
            command.add_argument('--feature-index',type=Path,required=True)
            command.add_argument('--stage',choices=('development',),default='development')
        if name=='summarize':
            command.add_argument('--batch',type=Path,required=True);command.add_argument('--output',type=Path)
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        paths=load_paths(args.paths)
        from .e1_cli import require_official_e1
        original=(args.batch/'parent/parent/parent/feature_index_snapshot.json' if args.command=='summarize'
                  else args.e12_batch/'parent/parent/feature_index_snapshot.json')
        require_official_e1(original)
        if args.command in ('check','prepare'):
            from .e13_workflow import check_parent,prepare_e13
            config=load_config(args.config)
            if args.command=='check':
                print(json.dumps(check_parent(args.e12_batch,config,paths),ensure_ascii=False,indent=2));return 0
            result=prepare_e13(args.e12_batch,config,paths,args.output,device=args.device)
        elif args.command=='run':
            from .e13_workflow import OFFICIAL,run_e13
            if read_json(args.feature_index).get('feature_condition')!=OFFICIAL:raise ValueError('production E1.3 requires official native class-text features')
            result=run_e13(args.e12_batch,args.feature_index,paths,stage=args.stage)
        else:
            from .e13_reporting import validate_runs,summarize
            from .e13_workflow import OFFICIAL
            batch,_=validate_runs(args.batch)
            if batch['feature_condition']!=OFFICIAL:raise ValueError('production E1.3 requires official class-text features')
            root=safe_output(args.output or paths['report_root'],paths,protected=(args.batch,))
            folder=new_directory(root,'e13-class-text-report');summarize(args.batch,folder);result=folder/'e13-results.md'
        print(result);return 0
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);return 2
