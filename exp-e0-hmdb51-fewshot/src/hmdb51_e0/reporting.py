"""Validate a complete run batch before computing across-support statistics."""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import numpy as np

from .config import CLASSES
from .metrics import classification_metrics
from .pipeline import load_bundle, official_clip_condition, protocol_config, smoke_ids
from .provenance import digest, read_json, sha256_file, write_csv, write_json
from .sampling import feasible_shots, validate_support_sets


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validated_runs(directory: Path, config: dict) -> tuple[dict, list[dict]]:
    directory = Path(directory)
    batch = read_json(directory / "batch.json")
    if batch["status"] != "success" or batch["errors"] or batch["config"] != protocol_config(config):
        raise ValueError("cannot aggregate incomplete/failed/mixed-config batch")
    bundle = load_bundle(directory / "manifest_snapshot.json", config)
    stage = batch["evaluation_stage"]
    if stage not in ("development", "smoke"):
        raise ValueError("final_test is sealed")
    expected_pairs = {(1, 0)} if stage == "smoke" else {
        (k, seed) for k in feasible_shots(bundle["rows"], config) for seed in config["sampling"]["support_seeds"]}
    actual_pairs = [(r["k"], r["support_seed"]) for r in batch["runs"]]
    if set(actual_pairs) != expected_pairs or len(actual_pairs) != len(expected_pairs) or {tuple(p) for p in batch["expected_pairs"]} != expected_pairs:
        raise ValueError("missing or duplicate seed/K runs")
    expected_evaluation = smoke_ids(bundle["rows"])[1] if stage == "smoke" else sorted(
        r["video_id"] for r in bundle["rows"] if r["experimental_role"] == "development")
    if batch["protocol_id"] != bundle["protocol_id"] or batch["evaluation_ids"] != expected_evaluation:
        raise ValueError("mixed protocol/evaluation identities")
    feature_index = read_json(directory / "feature_index_snapshot.json")
    if digest(feature_index["identity"]) != batch["feature_identity"]:
        raise ValueError("feature snapshot identity mismatch")
    condition = "official_openai_clip_vit_b32" if official_clip_condition(feature_index["identity"]) else "validation_only_unverified_features"
    if batch.get("feature_condition") != condition:
        raise ValueError("feature condition label mismatch")
    support_records = read_json(directory / "support_sets_snapshot.json")
    if stage == "development":
        validate_support_sets(support_records, bundle["rows"], config, bundle["protocol_id"])
    by_id = {r["video_id"]: r for r in bundle["rows"]}
    results = []
    for run in batch["runs"]:
        name = run["directory"]
        if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name or ":" in name:
            raise ValueError("unsafe run directory")
        run_dir = directory / name
        provenance = read_json(run_dir / "provenance.json")
        if provenance["status"] != "success" or provenance["run_id"] != name:
            raise ValueError("failed or inconsistent run")
        for key in ("protocol_id", "evaluation_stage", "feature_identity", "manifest_file_sha256", "evaluation_ids", "source"):
            if provenance[key] != batch[key]:
                raise ValueError(f"mixed run identity: {key}")
        if any(provenance[k] != run[k] for k in ("k", "support_seed")):
            raise ValueError("run K/seed mismatch")
        if read_json(run_dir / "resolved_config.json") != protocol_config(config):
            raise ValueError("mixed resolved config")
        for filename, key in (("predictions.csv", "predictions_sha256"), ("metrics.json", "metrics_sha256"), ("support_set.csv", "support_csv_sha256")):
            if sha256_file(run_dir / filename) != provenance[key]:
                raise ValueError(f"corrupt run artifact: {filename}")
        support = _read_csv(run_dir / "support_set.csv")
        support_ids = [r["video_id"] for r in support]
        expected_support = [r for r in support_records if r["k"] == run["k"] and r["support_seed"] == run["support_seed"]]
        if stage == "smoke":
            expected_support_ids = set(smoke_ids(bundle["rows"])[0])
        else:
            expected_support_ids = {r["video_id"] for r in expected_support}
        if len(support_ids) != 5 * run["k"] or set(support_ids) != expected_support_ids or support_ids != provenance["support_ids"]:
            raise ValueError("support identities/counts mismatch")
        if digest(sorted(support_ids)) != provenance["support_sha256"] or provenance["n_support"] != 5 * run["k"]:
            raise ValueError("support hash/count mismatch")
        predictions = _read_csv(run_dir / "predictions.csv")
        if [r["video_id"] for r in predictions] != expected_evaluation:
            raise ValueError("evaluation rows missing, duplicated, or inconsistent")
        try:
            truth = np.array([int(r["true_class_id"]) for r in predictions])
            scores = np.array([[float(r[f"score_{c}"]) for c in CLASSES] for r in predictions])
            claimed = np.array([int(r["predicted_class_id"]) for r in predictions])
            for r in predictions:
                if r["run_id"] != name or r["evaluation_stage"] != stage or int(r["k"]) != run["k"] or int(r["support_seed"]) != run["support_seed"]:
                    raise ValueError("prediction run identity mismatch")
            if truth.tolist() != [by_id[vid]["class_id"] for vid in expected_evaluation] or not np.array_equal(claimed, scores.argmax(axis=1)):
                raise ValueError("prediction label/argmax mismatch")
            recomputed = {**classification_metrics(truth, scores), "evaluation_stage": stage, "n_support": len(support_ids)}
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("malformed predictions") from exc
        metrics = read_json(run_dir / "metrics.json")
        if metrics != recomputed:
            raise ValueError("saved metrics do not match per-video predictions")
        results.append({**run, "metrics": metrics, "support_sha256": provenance["support_sha256"]})
    return batch, results


def summarize_batch(directory: Path, output: Path, config: dict) -> list[dict]:
    batch, runs = validated_runs(directory, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary, confusions = [], {}
    for k in sorted({r["k"] for r in runs}):
        group = [r for r in runs if r["k"] == k]
        row = {"evaluation_stage": batch["evaluation_stage"], "k": k, "n_support_sets": len(group),
               "n_unique_support_sets": len({r["support_sha256"] for r in group}),
               "n_evaluated_videos": group[0]["metrics"]["n_evaluated"]}
        for metric in ("accuracy", "macro_f1"):
            values = [r["metrics"][metric] for r in group]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        for i, name in enumerate(CLASSES):
            values = [r["metrics"]["per_class_recall"][i] for r in group]
            row[f"recall_{name}_mean"] = float(np.mean(values))
            row[f"recall_{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
        confusions[str(k)] = np.sum([r["metrics"]["confusion_matrix"] for r in group], axis=0).tolist()
    write_json(output / "summary.json", {"batch": str(Path(directory).resolve()), "protocol_id": batch["protocol_id"],
                                        "feature_identity": batch["feature_identity"], "summary": summary,
                                        "confusion_matrix_sum_over_support_sets": confusions})
    write_csv(output / "summary.csv", summary, list(summary[0]))
    _plot(summary, confusions, output)
    bundle = read_json(Path(directory) / "manifest_snapshot.json")
    counts = Counter((r["class_name"], r["experimental_role"]) for r in bundle["rows"])
    official = batch["feature_condition"] == "official_openai_clip_vit_b32"
    title = "# E0：HMDB51 冻结 CLIP 少样本开发结果" if official else "# 仅程序验证：非官方 CLIP 特征，不能作为 E0 实验结果"
    representation = "OpenAI CLIP ViT-B/32 冻结原生图像特征" if official else "未验证/合成特征，仅验证计算流程"
    text = [title, "",
            f"阶段：**{batch['evaluation_stage']}**；最终测试未开启。", "",
            f"固定五类 clap/jump/run/walk/wave；{representation}；16 个全视频中点帧，帧归一化后均值＋总体标准差；多输出岭分类，λ=1、截距不惩罚。", "",
            "| K | 支持集数 / 唯一数 | 每次评估视频 | Accuracy（均值 ± 样本标准差） | Macro-F1（均值 ± 样本标准差） |",
            "| --- | --- | --- | --- | --- |"]
    fmt = lambda value: "null" if value is None else f"{100 * value:.2f}%"
    for r in summary:
        text.append(f"| {r['k']} | {r['n_support_sets']} / {r['n_unique_support_sets']} | {r['n_evaluated_videos']} | {fmt(r['accuracy_mean'])} ± {fmt(r['accuracy_std'])} | {fmt(r['macro_f1_mean'])} ± {fmt(r['macro_f1_std'])} |")
    text += ["", "![学习曲线](learning_curve.png)", "", "| 类别 | 支持池 | 开发 | 封存测试 | 本折不用 | 排除 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name in CLASSES:
        text.append("| " + " | ".join([name, *[str(counts[name, role]) for role in ("support_pool", "development", "final_test", "unused", "excluded")]]) + " |")
    text += ["", "![混淆矩阵](confusion_matrices.png)", "",
             "混淆矩阵为各支持集合预测计数之和，同一开发视频被重复评价；不是更多独立测试视频。波动表示支持集选择敏感性；仅报告固定五类的开发表现，不代表全 51 类、标准 FSAR 或独立测试成绩。所有种子均保留，未按分数筛选。", "",
             "重复审计覆盖完整视频 SHA-256；尚未验证不同文件之间的原始来源片段分组。具有相似来源文件名的视频不能据此宣称来源互斥，后续若确认同源重复需按实验说明修订分组协议并保留本次结果。", "",
             f"协议身份：`{batch['protocol_id']}`。", "", f"特征身份：`{batch['feature_identity']}`。", "",
             f"原始运行目录：`{Path(directory).resolve()}`；其中包含逐视频预测、支持身份、环境、配置、来源快照和失败记录。", "",
             "CLIP 已接受图文预训练。后续语义实验衡量显式概念带来的额外收益。后续方法必须复用当前开发与支持 ID；最终测试需在全部比较方案冻结后另行开放。", ""]
    (output / "e0-results.md").write_text("\n".join(text), encoding="utf-8")
    return summary


def _plot(summary: list[dict], confusions: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    x = [r["k"] for r in summary]
    for name, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        ax.errorbar(x, [100 * r[f"{name}_mean"] for r in summary],
                    yerr=[100 * (r[f"{name}_std"] or 0) for r in summary], marker="o", capsize=4, label=label)
    ax.set(xlabel="Labeled videos per class (K)", ylabel="Development score (%)", xticks=x, ylim=(0, 100),
           title="E0 frozen CLIP — mean ± sample SD across support sets")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "learning_curve.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(summary), figsize=(5 * len(summary), 4.6), squeeze=False, layout="constrained")
    for ax, row in zip(axes[0], summary):
        matrix = np.array(confusions[str(row["k"])])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Predicted", ylabel="True", title=f"K={row['k']}; sum of {row['n_support_sets']} support sets")
    fig.savefig(output / "confusion_matrices.png", dpi=180)
    plt.close(fig)
