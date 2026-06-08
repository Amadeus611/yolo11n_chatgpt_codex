"""codex3 模块改进与轻量化筛选脚本。

该脚本只比较结构候选，不继续混入 SNDQ v1。默认输出到 runs/codex3。
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import torch  # type: ignore
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = "EVD4UAV.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "codex3"

EXP04_BEST_MAP50_95 = 0.49922
CODEX2_BEST_MAP50_95 = 0.50342
SCREEN_PASS_MAP50_95 = 0.49900
LIGHT_MAP50_95_FLOOR = EXP04_BEST_MAP50_95 - 0.003


@dataclass(frozen=True)
class Experiment:
    """单个 codex3 结构候选。"""

    name: str
    model: str
    desc: str
    epochs: int = 120
    imgsz: int = 640
    batch: int = 64
    extra: dict[str, Any] = field(default_factory=dict)

    def train_kwargs(self, args: argparse.Namespace, project: Path, final: bool) -> dict[str, Any]:
        """生成训练参数。"""
        name = f"Final_{self.name}" if final and not self.name.startswith("Final_") else self.name
        epochs = args.final_epochs if final else self.epochs
        kwargs: dict[str, Any] = dict(
            data=args.data,
            imgsz=self.imgsz,
            batch=self.batch,
            name=name,
            project=str(project),
            exist_ok=args.exist_ok,
            device=args.device,
            workers=args.workers,
            val=True,
            plots=True,
            save=True,
            amp=True,
            deterministic=False,
            cache=args.cache,
            pretrained=True,
            optimizer="SGD",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            cos_lr=True,
            warmup_epochs=5,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            cls_pw=0.3,
            sndq=False,
            epochs=epochs,
            patience=100,
            mosaic=0.8,
            close_mosaic=40,
            mixup=0.0,
            copy_paste=0.1,
            degrees=25.0,
            scale=0.25,
            translate=0.08,
            fliplr=0.5,
            flipud=0.5,
            erasing=0.0,
            hsv_h=0.01,
            hsv_s=0.4,
            hsv_v=0.3,
            dropout=0.0,
        )
        kwargs.update(self.extra)
        return kwargs


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        name="Codex3_C3Lite_Only",
        model="ultralytics/cfg/models/11/yolo11-codex3-c3lite-only.yaml",
        desc="保留 SCSPDRCF 和 C3k2Lite，取消 DySample，验证能否降低实际延迟。",
    ),
    Experiment(
        name="Codex3_DySample_Only",
        model="ultralytics/cfg/models/11/yolo11-codex3-dysample-only.yaml",
        desc="保留 SCSPDRCF 和 DySample，P3 融合恢复 C3k2，拆分 DySample 贡献。",
    ),
    Experiment(
        name="Codex3_LiteNeck_N8",
        model="ultralytics/cfg/models/11/yolo11-codex3-litenneck-n8.yaml",
        desc="C3k2Lite 的 n_div 从 4 调到 8，进一步减少空间卷积计算。",
    ),
    Experiment(
        name="Codex3_SCSP_Alpha005",
        model="ultralytics/cfg/models/11/yolo11-codex3-scsp-alpha005.yaml",
        desc="SCSPDRCF alpha 从 0.1 降到 0.05，减少浅层细节噪声反灌。",
    ),
    Experiment(
        name="Codex3_SCSP_Alpha015",
        model="ultralytics/cfg/models/11/yolo11-codex3-scsp-alpha015.yaml",
        desc="SCSPDRCF alpha 从 0.1 提到 0.15，验证小目标召回收益。",
    ),
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 codex3 模块改进与轻量化筛选实验。")
    parser.add_argument("--stage", choices=("screen", "final", "summary", "profile"), default="screen")
    parser.add_argument("--experiments", nargs="*", help="只运行指定实验名；final 阶段建议指定 1-2 个候选。")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="输出目录，默认 runs/codex3。")
    parser.add_argument("--data", default=DEFAULT_DATA, help="数据集 YAML。")
    parser.add_argument("--device", default="0", help="训练设备。")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers。")
    parser.add_argument("--cache", action="store_true", help="启用 Ultralytics 数据缓存。")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名实验目录。")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要运行的实验。")
    parser.add_argument("--final-epochs", type=int, default=300, help="final 阶段正式训练轮数。")
    return parser.parse_args()


def select_experiments(names: list[str] | None) -> list[Experiment]:
    """按实验名筛选。"""
    experiments = list(EXPERIMENTS)
    if not names:
        return experiments
    by_name = {exp.name: exp for exp in experiments}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"未知实验名: {', '.join(missing)}。可选: {', '.join(by_name)}")
    return [by_name[name] for name in names]


def warn_if_data_path_needs_check(data: str) -> None:
    """提示检查数据路径。"""
    data_path = Path(data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        print(f"[警告] 未找到数据配置文件: {data_path}")
        return
    text = data_path.read_text(encoding="utf-8", errors="ignore")
    if "/home/ssssss/1yolo/Dataset/EVD4UAV" in text:
        print("[提醒] EVD4UAV.yaml 仍是旧 Linux path，正式训练前请确认它在当前机器可访问。")


def run_experiments(args: argparse.Namespace) -> None:
    """执行训练实验。"""
    final = args.stage == "final"
    project = Path(args.project).resolve()
    experiments = select_experiments(args.experiments)
    warn_if_data_path_needs_check(args.data)

    for index, exp in enumerate(experiments, start=1):
        kwargs = exp.train_kwargs(args, project, final=final)
        print(f"\n{'=' * 80}")
        print(f"实验 {index}/{len(experiments)}: {kwargs['name']}")
        print(f"说明: {exp.desc}")
        print(f"模型: {exp.model}")
        print(f"输出: {project / kwargs['name']}")
        print(
            f"关键参数: epochs={kwargs['epochs']}, imgsz={kwargs['imgsz']}, batch={kwargs['batch']}, "
            f"mosaic={kwargs['mosaic']}, close_mosaic={kwargs['close_mosaic']}, sndq={kwargs['sndq']}"
        )
        print(f"{'=' * 80}\n")
        if args.dry_run:
            continue

        model = YOLO(exp.model)
        model.train(**kwargs)
        torch.cuda.empty_cache()


def best_metrics(results_csv: Path) -> dict[str, float] | None:
    """读取 results.csv 中的最佳指标。"""
    if not results_csv.exists():
        return None
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def get(row: dict[str, str], key: str) -> float:
        return float(row[key].strip())

    map50_key = "metrics/mAP50(B)"
    map95_key = "metrics/mAP50-95(B)"
    precision_key = "metrics/precision(B)"
    recall_key = "metrics/recall(B)"
    best_map95_row = max(rows, key=lambda row: get(row, map95_key))
    best_map50_row = max(rows, key=lambda row: get(row, map50_key))
    return {
        "best_epoch_mAP50": get(best_map50_row, "epoch"),
        "best_mAP50": get(best_map50_row, map50_key),
        "best_epoch_mAP50_95": get(best_map95_row, "epoch"),
        "best_mAP50_95": get(best_map95_row, map95_key),
        "precision_at_best_mAP95": get(best_map95_row, precision_key),
        "recall_at_best_mAP95": get(best_map95_row, recall_key),
    }


def summarize(project: Path) -> None:
    """汇总 codex3 筛选结果。"""
    print("\ncodex3 筛选标准:")
    print(f"- 精度候选: mAP50:95 >= {SCREEN_PASS_MAP50_95:.5f}")
    print(f"- 轻量候选: mAP50:95 >= {LIGHT_MAP50_95_FLOOR:.5f} 且实际速度或 GFLOPs 有收益")
    print(f"- codex2 当前最好 mAP50:95 = {CODEX2_BEST_MAP50_95:.5f}")
    print("\n实验汇总:")
    print("name,best_mAP50,best_mAP50_95,precision_at_best_mAP95,recall_at_best_mAP95,tag")

    for exp in EXPERIMENTS:
        metrics = best_metrics(project / exp.name / "results.csv")
        if metrics is None:
            print(f"{exp.name},未完成,未完成,未完成,未完成,NO")
            continue

        if metrics["best_mAP50_95"] >= SCREEN_PASS_MAP50_95:
            tag = "ACCURACY"
        elif metrics["best_mAP50_95"] >= LIGHT_MAP50_95_FLOOR:
            tag = "LIGHT_CHECK"
        else:
            tag = "NO"
        print(
            f"{exp.name},"
            f"{metrics['best_mAP50']:.5f},"
            f"{metrics['best_mAP50_95']:.5f},"
            f"{metrics['precision_at_best_mAP95']:.5f},"
            f"{metrics['recall_at_best_mAP95']:.5f},"
            f"{tag}"
        )


def profile_models(experiments: list[Experiment]) -> None:
    """打印候选模型的结构摘要。"""
    for exp in experiments:
        print(f"\n{'=' * 80}")
        print(f"模型: {exp.name}")
        print(f"配置: {exp.model}")
        model = YOLO(exp.model)
        model.info(detailed=False, verbose=True)


def main() -> None:
    """入口函数。"""
    args = parse_args()
    project = Path(args.project).resolve()
    experiments = select_experiments(args.experiments)

    if args.stage == "summary":
        summarize(project)
        return
    if args.stage == "profile":
        profile_models(experiments)
        return

    run_experiments(args)
    if not args.dry_run:
        summarize(project)


if __name__ == "__main__":
    main()
