"""Production CLI for the fixed-person, fixed-frame ROI experiment."""
import argparse,json,sys
from pathlib import Path
from .config import ROOT,load_paths
from .e20r2_protocol import load_config
from .e20r2_workflow import STAGES
from .provenance import new_directory
from .round2_io import safe_output

def build_parser():
    parser=argparse.ArgumentParser(description='E2.0-R2 fixed60 frames/person ROI; frozen CLIP; development only')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('check','prepare','status','freeze','encode','summarize'):
        p=sub.add_parser(name);p.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
        if name in ('check','prepare'):
            p.add_argument('--parent',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/e20r2.person-roi.yaml')
        else:p.add_argument('--packet',type=Path,required=True)
        if name=='prepare':p.add_argument('--output',type=Path,default=ROOT/'data/e20r2_audits')
        if name=='freeze':p.add_argument('--view',choices=STAGES,required=True)
        if name=='encode':p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
        if name=='summarize':p.add_argument('--output',type=Path)
    return parser

def main(argv=None):
    a=build_parser().parse_args(argv)
    try:
        from .e20r2_workflow import check_inputs,prepare,load_packet,freeze,encode
        paths=load_paths(a.paths)
        if a.command in ('check','prepare'):
            config=load_config(a.config)
            if a.command=='check':print(json.dumps(check_inputs(a.parent,config),ensure_ascii=False,indent=2));return 0
            result=prepare(a.parent,paths,a.output,config)
        else:
            c=load_packet(a.packet,require_official=True)
            if a.command=='status':
                print(json.dumps(dict(packet=str(c['packet']),encoded=c['manifest']['encoding'] is not None,
                    frozen=list(c['manifest']['locks']),progress={v:dict(n_rows=len(rows),n_completed=sum(bool(r['target_status' if v=='targets' else 'judgment']) for r in rows))
                        for v,rows in c['forms'].items()}),ensure_ascii=False,indent=2));return 0
            if a.command=='freeze':result=freeze(a.packet,a.view)
            elif a.command=='encode':result=encode(a.packet,paths,device=a.device)
            else:
                from .e20r2_reporting import summarize
                root=safe_output(a.output or paths['report_root'],paths,protected=(a.packet,c['source']['parent'],c['source']['e11_batch'],c['source']['audit_directory']))
                folder=new_directory(root,'e20r2-person-roi-report');summarize(a.packet,folder);result=folder/'e20r2-results.md'
        print(result);return 0
    except (ValueError,OSError,KeyError,TypeError,IndexError,RuntimeError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);return 2
