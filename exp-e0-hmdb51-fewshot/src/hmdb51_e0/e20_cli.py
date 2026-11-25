"""Explicit E2.0 preflight, packet preparation, human freezes and reporting."""
import argparse
import json
from pathlib import Path
import sys

from .config import ROOT, load_paths
from .e20_protocol import load_config, VIEWS
from .provenance import new_directory
from .round2_io import safe_output


def build_parser():
    parser = argparse.ArgumentParser(description='E2.0 visible concept audit; no model training or automatic labels')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'prepare', 'status', 'freeze', 'summarize'):
        command = sub.add_parser(name)
        command.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
        if name in ('check', 'prepare'):
            command.add_argument('--e11-batch', type=Path, required=True)
            command.add_argument('--audit-directory', type=Path, required=True)
            command.add_argument('--config', type=Path, default=ROOT/'configs/e20.concept-grounding.yaml')
        else:
            command.add_argument('--packet', type=Path, required=True)
        if name == 'prepare':
            command.add_argument('--output', type=Path, default=ROOT/'data/e20_audits')
        if name == 'freeze':
            command.add_argument('--view', choices=VIEWS, required=True)
        if name == 'summarize':
            command.add_argument('--output', type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        paths = load_paths(args.paths)
        from .e20_workflow import check_inputs, prepare_e20, load_packet, freeze_view
        if args.command in ('check', 'prepare'):
            config = load_config(args.config)
            if args.command == 'check':
                print(json.dumps(check_inputs(args.e11_batch, args.audit_directory, config, require_official=True), ensure_ascii=False, indent=2))
                return 0
            result = prepare_e20(args.e11_batch, args.audit_directory, config, paths, args.output, require_official=True)
        else:
            context = load_packet(args.packet, require_official=True)
            if args.command == 'status':
                print(json.dumps(dict(packet=str(context['packet']), frozen_views=list(context['manifest']['locks']),
                    progress={v: dict(n_rows=len(rows), n_judged=sum(bool(r['judgment'].strip()) for r in rows))
                              for v, rows in context['forms'].items()}), ensure_ascii=False, indent=2))
                return 0
            if args.command == 'freeze':
                result = freeze_view(args.packet, args.view)
            else:
                from .e20_reporting import summarize
                root = safe_output(args.output or paths['report_root'], paths,
                    protected=(args.packet, context['manifest']['e11_batch'], context['manifest']['audit_directory']))
                folder = new_directory(root, 'e20-grounding-report')
                summarize(args.packet, folder)
                result = folder/'e20-results.md'
        print(result)
        return 0
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
