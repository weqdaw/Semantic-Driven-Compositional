"""Separate read-only checks, derivation, human-audit export, fitting and reporting."""
import argparse
import json
from pathlib import Path
import sys

from .config import ROOT, load_paths
from .e11_features import load_config
from .provenance import new_directory
from .round2_io import safe_output


def build_parser():
    parser=argparse.ArgumentParser(description='E1.1 G1 / VC1 / VG1 diagnostics; final test sealed')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('check','prepare','export-audit','run','summarize'):
        command=sub.add_parser(name)
        command.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
        if name!='summarize': command.add_argument('--e1-batch',type=Path,required=True)
        if name in ('check','prepare','export-audit'):
            command.add_argument('--config',type=Path,default=ROOT/'configs/e11.diagnostics.yaml')
        if name=='prepare': command.add_argument('--output',type=Path,default=ROOT/'artifacts/e11_diagnostics')
        if name=='export-audit': command.add_argument('--output',type=Path,default=ROOT/'data/e11_audits')
        if name=='run':
            command.add_argument('--feature-index',type=Path,required=True)
            command.add_argument('--stage',choices=('development',),default='development')
        if name=='summarize':
            command.add_argument('--batch',type=Path,required=True)
            command.add_argument('--output',type=Path,help='Report parent; a new child is created')
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        paths=load_paths(args.paths)
        from .e1_cli import require_official_e1
        official_index=(args.batch/'parent/feature_index_snapshot.json' if args.command=='summarize'
                        else args.e1_batch/'feature_index_snapshot.json')
        require_official_e1(official_index)
        if args.command in ('check','prepare','export-audit'):
            config=load_config(args.config)
            if args.command=='check':
                from .e11_workflow import check_parent
                print(json.dumps(check_parent(args.e1_batch,config),ensure_ascii=False,indent=2)); return 0
            if args.command=='prepare':
                from .e11_workflow import prepare_features
                result=prepare_features(args.e1_batch,config,paths,args.output)
            else:
                from .e11_audit import export_audit
                result=export_audit(args.e1_batch,config,paths,args.output)
        elif args.command=='run':
            from .e11_workflow import run_e11
            result=run_e11(args.e1_batch,args.feature_index,paths,stage=args.stage)
        else:
            from .e11_reporting import summarize,validate_runs
            validate_runs(args.batch)
            parent=safe_output(args.output or paths['report_root'],paths,protected=(args.batch,))
            folder=new_directory(parent,'e11-diagnostic-report')
            summarize(args.batch,folder); result=folder/'e11-results.md'
        print(result); return 0
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr); return 2
