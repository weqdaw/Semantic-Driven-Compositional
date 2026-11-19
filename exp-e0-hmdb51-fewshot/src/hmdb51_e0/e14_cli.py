"""Explicit E1.4 read-only checks, snapshot preparation, evaluation and reports."""
import argparse
import json
from pathlib import Path
import sys

from .config import ROOT,load_paths
from .e14_features import load_config
from .e14_workflow import OFFICIAL
from .provenance import read_json,new_directory
from .round2_io import safe_output


def build_parser():
    parser=argparse.ArgumentParser(description='E1.4 fixed text anchor,80 new evaluations and21 references')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('check','prepare','run','summarize'):
        command=sub.add_parser(name);command.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
        if name!='summarize':command.add_argument('--e13-batch',type=Path,required=True)
        if name in ('check','prepare'):command.add_argument('--config',type=Path,default=ROOT/'configs/e14.text-anchor.yaml')
        if name=='prepare':command.add_argument('--output',type=Path,default=ROOT/'artifacts/e14_text_anchor')
        if name=='run':
            command.add_argument('--feature-index',type=Path,required=True);command.add_argument('--stage',choices=('development',),default='development')
        if name=='summarize':
            command.add_argument('--batch',type=Path,required=True);command.add_argument('--output',type=Path)
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        paths=load_paths(args.paths)
        parent_index=args.batch/'parent/feature_index_snapshot.json' if args.command=='summarize' else args.e13_batch/'feature_index_snapshot.json'
        if read_json(parent_index).get('feature_condition')!=OFFICIAL:raise ValueError('production E1.4 requires official E1.3 class-text features')
        if args.command in ('check','prepare'):
            from .e14_workflow import check_parent,prepare_e14
            config=load_config(args.config)
            if args.command=='check':print(json.dumps(check_parent(args.e13_batch,config),ensure_ascii=False,indent=2));return 0
            result=prepare_e14(args.e13_batch,config,paths,args.output)
        elif args.command=='run':
            from .e14_workflow import run_e14
            result=run_e14(args.e13_batch,args.feature_index,paths,stage=args.stage)
        else:
            from .e14_reporting import validate_runs,summarize
            validate_runs(args.batch)
            root=safe_output(args.output or paths['report_root'],paths,protected=(args.batch,));folder=new_directory(root,'e14-text-anchor-report')
            summarize(args.batch,folder);result=folder/'e14-results.md'
        print(result);return 0
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);return 2
