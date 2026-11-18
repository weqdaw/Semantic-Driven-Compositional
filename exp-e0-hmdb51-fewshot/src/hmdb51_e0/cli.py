"""Six explicit stages with nonzero failure exits and no implicit test opening."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import ROOT, load_config, load_paths
from .provenance import new_directory, read_json, write_json


def require_official_features(index_path: Path) -> None:
    from .pipeline import official_clip_condition
    if not official_clip_condition(read_json(index_path).get("identity", {})):
        raise ValueError("production E0 requires the official OpenAI CLIP ViT-B/32 feature identity")


def main(stage: int, argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"HMDB51 E0 stage {stage}; final test remains sealed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e0.v0.example.yaml")
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.local.yaml")
    if stage == 1:
        parser.add_argument("--aliases", type=Path, help="explicit official-content-verified path aliases JSON")
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--output", type=Path, default=ROOT / "data/manifests")
    elif stage == 2:
        parser.add_argument("--audit", type=Path, required=True)
        parser.add_argument("--output", type=Path, default=ROOT / "data/manifests")
    elif stage in (3, 4, 5):
        parser.add_argument("--manifest", type=Path, required=True, help="frozen bundle.json")
        if stage in (3, 5):
            parser.add_argument("--stage", choices=("smoke", "development"), default="development")
        if stage == 3:
            parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
            parser.add_argument("--batch-size", type=int, default=16)
        elif stage == 4:
            parser.add_argument("--output", type=Path, default=ROOT / "data/support_sets")
        else:
            parser.add_argument("--support-sets", type=Path, required=True)
            parser.add_argument("--feature-index", type=Path, required=True)
    elif stage == 6:
        parser.add_argument("--batch", type=Path, required=True)
        parser.add_argument("--output", type=Path)
    else:
        raise ValueError("unknown stage")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        paths = load_paths(args.paths)
        if stage == 1:
            from .inventory import audit_dataset
            if "annotations_root" not in paths:
                raise ValueError("annotations_root is missing; import official annotations first")
            result = audit_dataset(paths["dataset_root"], paths["annotations_root"], workers=args.workers,
                                   aliases=read_json(args.aliases) if args.aliases else None)
            source = paths["annotations_root"].parent / "source.json"
            if source.is_file():
                result["annotation_source"] = read_json(source)
            output = new_directory(args.output, "audit") / "audit.json"
            write_json(output, result)
            print(output)
            if result["status"] != "success":
                print("BLOCKED: " + "; ".join(result["errors"]), file=sys.stderr)
                return 2
        elif stage == 2:
            from .pipeline import freeze_manifest
            print(freeze_manifest(read_json(args.audit), config, args.output))
        elif stage == 3:
            from .clip_features import FrozenCLIP
            from .pipeline import extract_features
            if "weights_file" not in paths:
                raise ValueError("weights_file is missing; obtain the verified official checkpoint")
            encoder = FrozenCLIP(paths["weights_file"], args.device, args.batch_size)
            print(extract_features(args.manifest, config, paths, encoder, stage=args.stage))
        elif stage == 4:
            from .pipeline import load_bundle, write_supports
            print(write_supports(load_bundle(args.manifest, config), config, args.output))
        elif stage == 5:
            from .pipeline import run_development
            require_official_features(args.feature_index)
            print(run_development(args.manifest, args.support_sets, args.feature_index, config, paths, stage=args.stage))
        else:
            from .reporting import summarize_batch
            output = args.output or new_directory(paths["report_root"], "e0-v0-report")
            summarize_batch(args.batch, output, config)
            print(output / "e0-results.md")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
