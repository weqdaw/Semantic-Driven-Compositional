"""Explicit E1.5 read-only preflight, snapshots, evaluation and reporting."""
import argparse
import json
from pathlib import Path
import sys

from .config import ROOT, load_paths
from .e13_workflow import OFFICIAL
from .e15_features import load_config
from .provenance import read_json, new_directory
from .round2_io import safe_output


def build_parser():
    parser = argparse.ArgumentParser(description='E1.5 instance caches:30 new evaluations and11 fixed references')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'prepare', 'run', 'summarize'):
        command = sub.add_parser(name)
        command.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
        if name != 'summarize':
            command.add_argument('--e14-batch', type=Path, required=True)
        if name in ('check', 'prepare'):
            command.add_argument('--config', type=Path, default=ROOT/'configs/e15.instance-cache.yaml')
        if name == 'prepare':
            command.add_argument('--output', type=Path, default=ROOT/'artifacts/e15_instance_cache')
        if name == 'run':
            command.add_argument('--feature-index', type=Path, required=True)
            command.add_argument('--stage', choices=('development',), default='development')
        if name == 'summarize':
            command.add_argument('--batch', type=Path, required=True)
            command.add_argument('--output', type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        paths = load_paths(args.paths)
        parent_index = args.batch/'parent/feature_index_snapshot.json' if args.command == 'summarize' else args.e14_batch/'feature_index_snapshot.json'
        if read_json(parent_index).get('feature_condition') != OFFICIAL:
            raise ValueError('production E1.5 requires official validated E1.4 class-text features')
        if args.command in ('check', 'prepare'):
            from .e15_workflow import check_parent, prepare_e15
            config = load_config(args.config)
            if args.command == 'check':
                print(json.dumps(check_parent(args.e14_batch, config), ensure_ascii=False, indent=2))
                return 0
            result = prepare_e15(args.e14_batch, config, paths, args.output)
        elif args.command == 'run':
            from .e15_workflow import run_e15
            result = run_e15(args.e14_batch, args.feature_index, paths, stage=args.stage)
        else:
            from .e15_reporting import validate_runs, summarize
            validate_runs(args.batch)
            root = safe_output(args.output or paths['report_root'], paths, protected=(args.batch,))
            folder = new_directory(root, 'e15-instance-cache-report')
            summarize(args.batch, folder)
            result = folder/'e15-results.md'
        print(result)
        return 0
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
