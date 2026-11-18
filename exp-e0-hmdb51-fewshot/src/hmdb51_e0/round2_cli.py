"""Explicit source review, grouped preparation, paired fitting, and reporting."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import ROOT, load_paths
from .provenance import new_directory
from .round2_protocol import load_round2_config

DEFAULT_FEATURE_ROOT = ROOT / 'artifacts/round2_features'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='E0 round2 grouped V0/V1; final test remains sealed. No command runs implicitly.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    help_text = {
        'audit-sources': 'Generate UNRESOLVED training-only filename candidates for evidence review; no fitting',
        'freeze-split': 'Validate completed review and freeze a deterministic group-disjoint development split',
        'sample-supports': 'Freeze nested video-shot supports with source coverage; no fitting',
        'prepare-features': 'Verify raw video hashes and import saved Z; compute V0/V1 without CLIP encoding',
        'run': 'Fit BOTH V0 and V1 on the same split/supports; development only',
        'summarize': 'Validate all 60 runs and recompute paired development metrics',
    }
    for name, help_message in help_text.items():
        sub = subparsers.add_parser(name, help=help_message, description=help_message)
        sub.add_argument('--config', type=Path, default=ROOT / 'configs/e0.round2.example.yaml')
        sub.add_argument('--paths', type=Path, default=ROOT / 'configs/paths.local.yaml')
        if name in ('audit-sources', 'freeze-split'):
            sub.add_argument('--parent-manifest', type=Path, required=True, help='Existing first-round E0 bundle or manifest snapshot')
        if name == 'audit-sources':
            sub.add_argument('--output', type=Path, default=ROOT / 'data/source_groups', help='Parent directory; create a new source-audit directory')
        elif name == 'freeze-split':
            sub.add_argument('--source-review', type=Path, required=True, help='Evidence-reviewed UTF-8 source_groups.csv; explicitly identify AI-assisted reviewers')
            sub.add_argument('--output', type=Path, default=ROOT / 'data/manifests/round2')
        elif name in ('sample-supports', 'prepare-features', 'run'):
            sub.add_argument('--manifest', type=Path, required=True, help='Round2 grouped bundle.json, not the old E0 manifest')
            if name == 'sample-supports':
                sub.add_argument('--output', type=Path, default=ROOT / 'data/support_sets/round2')
            elif name == 'prepare-features':
                sub.add_argument('--parent-feature-index', type=Path, required=True, help='Existing first-round full development index or its batch snapshot')
                sub.add_argument('--output', type=Path, default=DEFAULT_FEATURE_ROOT, help='Separate derived feature root; never the old feature root')
            else:
                sub.add_argument('--support-sets', type=Path, required=True)
                sub.add_argument('--feature-index', type=Path, required=True, help='Prepared round2 index-development.json')
                sub.add_argument('--feature-root', type=Path, default=DEFAULT_FEATURE_ROOT)
                sub.add_argument('--stage', choices=('development',), default='development')
        else:
            sub.add_argument('--batch', type=Path, required=True, help='A complete round2 batch directory')
            sub.add_argument('--output', type=Path, help='Report parent; always creates a fresh e0-round2-report directory')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_round2_config(args.config)
        paths = load_paths(args.paths)
        from .round2_io import safe_output
        if args.command == 'audit-sources':
            from .round2_protocol import load_parent
            from .source_groups import audit_sources
            output = safe_output(args.output, paths, protected=(args.parent_manifest.parent,))
            result = audit_sources(load_parent(args.parent_manifest), output)
        elif args.command == 'freeze-split':
            from .round2_protocol import build_grouped_bundle, freeze_grouped_bundle, load_parent
            from .source_groups import read_review
            output = safe_output(args.output, paths, protected=(args.parent_manifest.parent,))
            bundle = build_grouped_bundle(load_parent(args.parent_manifest), read_review(args.source_review), config)
            result = freeze_grouped_bundle(bundle, output)
        elif args.command == 'sample-supports':
            from .round2_protocol import load_grouped_bundle, write_round2_supports
            output = safe_output(args.output, paths, protected=(args.manifest.parent,))
            result = write_round2_supports(load_grouped_bundle(args.manifest, config), config, output)
        elif args.command == 'prepare-features':
            from .cli import require_official_features
            from .round2_features import prepare_round2_features
            require_official_features(args.parent_feature_index)
            result = prepare_round2_features(args.manifest, args.parent_feature_index, config, paths, args.output)
        elif args.command == 'run':
            from .cli import require_official_features
            from .round2_runner import run_round2
            require_official_features(args.feature_index)
            result = run_round2(args.manifest, args.support_sets, args.feature_index, config, paths, args.feature_root, stage=args.stage)
        else:
            from .round2_reporting import summarize_round2, validated_round2_runs
            parent = safe_output(args.output or paths['report_root'], paths, protected=(args.batch,))
            # Validate before creating an output directory; a failed batch is not a report.
            validated_round2_runs(args.batch, config)
            output = new_directory(parent, 'e0-round2-report')
            summarize_round2(args.batch, output, config)
            result = output / 'round2-results.md'
        print(result)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
