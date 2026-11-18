"""Explicit read-only check, preparation, fitting and reporting for controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .ablation_features import load_ablation_config
from .config import ROOT, load_paths
from .provenance import new_directory
from .round2_io import safe_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='E0 supplementary controls; no implicit fitting; final test sealed')
    commands = parser.add_subparsers(dest='command', required=True)
    for command, help_text in (('check', 'Read-only: validate reference and cached frames; no controls or fitting'),
                               ('prepare', 'Compute repeat/shuffle features from cached Z; no fitting'),
                               ('run', 'Fit all repeat/shuffle controls on frozen supports; development only'),
                               ('summarize', 'Validate complete batch and recompute paired metrics')):
        sub = commands.add_parser(command, help=help_text, description=help_text)
        sub.add_argument('--paths', type=Path, default=ROOT / 'configs/paths.local.yaml')
        if command in ('check', 'prepare'):
            sub.add_argument('--config', type=Path, default=ROOT / 'configs/e0.round2.ablations.yaml')
            sub.add_argument('--reference-batch', type=Path, required=True)
            sub.add_argument('--feature-root', type=Path, default=ROOT / 'artifacts/round2_features')
            if command == 'prepare':
                sub.add_argument('--output', type=Path, default=ROOT / 'artifacts/round2_ablations')
        elif command == 'run':
            sub.add_argument('--reference-batch', type=Path, required=True)
            sub.add_argument('--feature-index', type=Path, required=True)
            sub.add_argument('--stage', choices=('development',), default='development')
        else:
            sub.add_argument('--batch', type=Path, required=True)
            sub.add_argument('--output', type=Path, help='Report parent; a fresh child directory is created')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = load_paths(args.paths)
        if args.command in ('check', 'prepare', 'run'):
            from .cli import require_official_features
            require_official_features(args.reference_batch / 'feature_index_snapshot.json')
        if args.command in ('check', 'prepare'):
            from .ablation_experiment import check_reference, prepare_ablation_features
            config = load_ablation_config(args.config)
            if args.command == 'check':
                print(json.dumps(check_reference(args.reference_batch, config, args.feature_root), ensure_ascii=False, indent=2))
                return 0
            result = prepare_ablation_features(args.reference_batch, config, paths, args.feature_root, args.output)
        elif args.command == 'run':
            from .ablation_experiment import run_ablations
            require_official_features(args.feature_index)
            result = run_ablations(args.reference_batch, args.feature_index, paths, stage=args.stage)
        else:
            from .ablation_reporting import summarize_ablations, validate_ablation_runs
            validate_ablation_runs(args.batch)
            parent = safe_output(args.output or paths['report_root'], paths, protected=(args.batch,))
            result = new_directory(parent, 'e0-round2-ablation-report')
            summarize_ablations(args.batch, result)
            result = result / 'ablation-results.md'
        print(result)
        return 0
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
