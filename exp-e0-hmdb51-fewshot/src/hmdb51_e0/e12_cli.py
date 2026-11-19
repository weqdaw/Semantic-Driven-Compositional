"""Explicit E1.2 stages, with official-only production and sealed final test."""
import argparse
import json
from pathlib import Path
import sys

from .config import ROOT,load_paths
from .e12_features import load_config
from .provenance import new_directory
from .round2_io import safe_output


def build_parser():
    parser=argparse.ArgumentParser(description='E1.2 support-only scale matching; fixed 3-shot / 240 fits')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('check','prepare','audit','run','summarize'):
        sub=commands.add_parser(name)
        sub.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
        if name!='summarize': sub.add_argument('--e11-batch',type=Path,required=True)
        if name in ('check','prepare'): sub.add_argument('--config',type=Path,default=ROOT/'configs/e12.support-scale.yaml')
        if name=='prepare': sub.add_argument('--output',type=Path,default=ROOT/'artifacts/e12_support_scale')
        if name in ('audit','summarize'):
            sub.add_argument('--audit-directory',type=Path,required=name=='audit')
            sub.add_argument('--output',type=Path,help='Report parent; a new child directory is created')
        if name=='run':
            sub.add_argument('--feature-index',type=Path,required=True)
            sub.add_argument('--stage',choices=('development',),default='development')
        if name=='summarize': sub.add_argument('--batch',type=Path,required=True)
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        paths=load_paths(args.paths)
        from .e1_cli import require_official_e1
        official_index=(args.batch/'parent/parent/feature_index_snapshot.json' if args.command=='summarize'
                        else args.e11_batch/'parent/feature_index_snapshot.json')
        require_official_e1(official_index)
        if args.command in ('check','prepare'):
            from .e12_workflow import check_parent,prepare_e12
            config=load_config(args.config)
            if args.command=='check':
                print(json.dumps(check_parent(args.e11_batch,config),ensure_ascii=False,indent=2)); return 0
            result=prepare_e12(args.e11_batch,config,paths,args.output)
        elif args.command=='run':
            from .e12_workflow import run_e12
            result=run_e12(args.e11_batch,args.feature_index,paths,stage=args.stage)
        elif args.command=='audit':
            from .e12_audit import analyze_audit,write_audit_report
            result=analyze_audit(args.audit_directory,args.e11_batch)
            root=safe_output(args.output or paths['report_root'],paths,protected=(args.e11_batch,args.audit_directory))
            folder=new_directory(root,'e12-audit-report'); write_audit_report(result,folder)
            result=folder/'audit-summary.md'
        else:
            from .e12_reporting import summarize,validate_runs
            validate_runs(args.batch)
            protected=(args.batch,) if args.audit_directory is None else (args.batch,args.audit_directory)
            root=safe_output(args.output or paths['report_root'],paths,protected=protected)
            folder=new_directory(root,'e12-support-scale-report')
            summarize(args.batch,folder,audit_directory=args.audit_directory); result=folder/'e12-results.md'
        print(result); return 0
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr); return 2
