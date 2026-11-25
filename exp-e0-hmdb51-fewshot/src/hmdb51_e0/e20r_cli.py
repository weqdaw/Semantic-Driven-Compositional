"""Explicit repair planning, decoding, encoding and review commands."""
import argparse
import json
from pathlib import Path
import sys
from .config import ROOT, load_paths
from .e20r_protocol import load_config, VIEWS
from .provenance import new_directory
from .round2_io import safe_output


def build_parser():
    parser=argparse.ArgumentParser(description='E2.0-R coverage repair; development only; frozen CLIP')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('check','plan','prepare','encode','status','freeze','summarize'):
        p=sub.add_parser(name)
        p.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
        if name in ('check','plan'):
            p.add_argument('--e11-batch',type=Path,required=True)
            p.add_argument('--audit-directory',type=Path,required=True)
            p.add_argument('--config',type=Path,default=ROOT/'configs/e20r.coverage-repair.yaml')
        elif name=='prepare':p.add_argument('--plan',type=Path,required=True)
        else:p.add_argument('--packet',type=Path,required=True)
        if name=='plan':
            p.add_argument('--output',type=Path,default=ROOT/'data/e20r_plans')
            p.add_argument('--supplement-selection',type=Path)
            p.add_argument('--base-plan',type=Path)
        if name=='prepare':p.add_argument('--output',type=Path,default=ROOT/'data/e20r_audits')
        if name=='encode':p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
        if name=='freeze':p.add_argument('--view',choices=VIEWS,required=True)
        if name=='summarize':p.add_argument('--output',type=Path)
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        from .e20r_workflow import check_inputs,plan_e20r,prepare_e20r,load_packet,encode_e20r,freeze_view
        paths=load_paths(args.paths)
        if args.command in ('check','plan'):
            config=load_config(args.config)
            checked=check_inputs(args.e11_batch,args.audit_directory,config,require_official=True)
            if args.command=='check':
                print(json.dumps(checked,ensure_ascii=False,indent=2));return 0
            result=plan_e20r(args.e11_batch,args.audit_directory,config,paths,args.output,
                selection_path=args.supplement_selection,base_plan=args.base_plan,require_official=True)
        elif args.command=='prepare':
            result=prepare_e20r(args.plan,paths,args.output,require_official=True)
        else:
            context=load_packet(args.packet,require_official=True)
            if args.command=='status':
                print(json.dumps(dict(cohort=context['manifest']['cohort'],
                    encoded=context['manifest']['encoding'] is not None,frozen_views=list(context['manifest']['locks']),
                    progress={v:dict(n_rows=len(rows),n_judged=sum(bool(r['judgment'].strip()) for r in rows))
                              for v,rows in context['forms'].items()}),ensure_ascii=False,indent=2));return 0
            if args.command=='encode':result=encode_e20r(args.packet,paths,device=args.device,require_official=True)
            elif args.command=='freeze':result=freeze_view(args.packet,args.view)
            else:
                from .e20r_reporting import summarize
                root=safe_output(args.output or paths['report_root'],paths,
                    protected=(args.packet,context['source']['e11_batch'],context['source']['audit_directory']))
                folder=new_directory(root,'e20r-coverage-report')
                summarize(args.packet,folder);result=folder/'e20r-results.md'
        print(result);return 0
    except (ValueError,OSError,KeyError,TypeError,IndexError,RuntimeError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);return 2
