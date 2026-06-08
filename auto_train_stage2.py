"""第二轮 YOLO11n 改进实验脚本。

默认执行 codex2 版本的 120 epoch 筛选实验，不覆盖第一轮 runs/codex1 结果。
正式训练时先运行本脚本的 summary，再用 --stage final --experiments 指定胜出的候选。
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
DEFAULT_PROJECT = ROOT / "runs" / "codex2"
EXP04_BEST = ROOT / "runs" / "codex1" / "Exp04_SCSP_DRCF_LiteNeck" / "weights" / "best.pt"

EXP04_EPOCH120 = {
    "mAP50": 0.59625,
    "mAP50_95": 0.47281,
    "precision": 0.65720,
}
SCREEN_PASS_MAP50 = 0.60100
SCREEN_PASS_MAP50_95 = 0.47600
SCREEN_MIN_PRECISION = EXP04_EPOCH120["precision"] - 0.02


@dataclass(frozen=True)
class Experiment:
    """单个二阶段实验配置。"""

    name: str
    model: str | Path
    desc: str
    epochs: int = 120
    imgsz: int = 640
    batch: int = 64
    lr0: float = 0.001
    lrf: float = 0.01
    warmup_epochs: int = 5
    mosaic: float = 0.8
    close_mosaic: int = 40
    degrees: float = 25.0
    sndq: bool = False
    sndq_gamma: float = 0.05
    sndq_margin: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def train_kwargs(self, args: argparse.Namespace, project: Path, final: bool) -> dict[str, Any]:
        """生成 Ultralytics 训练参数。"""
        name = f"Final_{self.name}" if final and not self.name.startswith("Final_") else self.name
        epochs = self.epochs
        if final and "Finetune" not in self.name:
            epochs = args.final_epochs

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
            lr0=self.lr0,
            lrf=self.lrf,
            momentum=0.937,
            weight_decay=0.0005,
            cos_lr=True,
            warmup_epochs=self.warmup_epochs,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            cls_pw=0.3,
            sndq=self.sndq,
            sndq_gamma=self.sndq_gamma,
            sndq_tau=32.0,
            sndq_c=12.8,
            sndq_kappa=64.0,
            sndq_margin=self.sndq_margin,
            epochs=epochs,
            patience=100,
            mosaic=self.mosaic,
            close_mosaic=self.close_mosaic,
            mixup=0.0,
            copy_paste=0.1,
            degrees=self.degrees,
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


SCREEN_EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        name="Stage2_Exp04_AugTune_Deg20",
        model="ultralytics/cfg/models/11/yolo11n-scsp-drcf-litenneck.yaml",
        desc="保持 exp4 结构，将 degrees 从 25 降到 20，验证 HBB 旋转增强是否压低 AP95。",
        degrees=20.0,
    ),
    Experiment(
        name="Stage2_Exp04_Finetune_NoMosaic",
        model=EXP04_BEST,
        desc="从 exp4 best.pt 低学习率微调，关闭 mosaic，优先冲 mAP50:95。",
        epochs=80,
        lr0=0.0003,
        lrf=0.05,
        warmup_epochs=2,
        mosaic=0.0,
        close_mosaic=0,
    ),
    Experiment(
        name="Stage2_Exp04_Img768",
        model="ultralytics/cfg/models/11/yolo11n-scsp-drcf-litenneck.yaml",
        desc="保持 exp4 结构，输入分辨率升到 768，验证小目标收益。",
        imgsz=768,
        batch=32,
    ),
    Experiment(
        name="Stage2_Exp03_SCSP_DRCF_SNDQ",
        model="ultralytics/cfg/models/11/yolo11n-scsp-drcf-sndq.yaml",
        desc="SCSP+DRCF 加 SNDQ、不加 LiteNeck，拆分 SNDQ 与 LiteNeck 的组合影响。",
        sndq=True,
    ),
    Experiment(
        name="Stage2_Full_SNDQ_Weak",
        model="ultralytics/cfg/models/11/yolo11n-scsp-drcf-sndq-litenneck.yaml",
        desc="完整结构弱化 SNDQ，验证 exp5 是否因为邻域惩罚过强而退步。",
        sndq=True,
        sndq_gamma=0.02,
        sndq_margin=0.05,
    ),
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行第二轮 YOLO11n 筛选与正式实验。")
    parser.add_argument("--stage", choices=("screen", "final", "summary"), default="screen", help="实验阶段。")
    parser.add_argument("--experiments", nargs="*", help="只运行指定实验名；final 阶段必须指定。")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="输出目录，默认 runs/codex2。")
    parser.add_argument("--data", default=DEFAULT_DATA, help="数据集 YAML。")
    parser.add_argument("--device", default="0", help="训练设备。")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers。")
    parser.add_argument("--cache", action="store_true", help="启用 Ultralytics 数据缓存。")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名实验目录。")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要运行的实验，不启动训练。")
    parser.add_argument("--final-epochs", type=int, default=300, help="final 阶段正式训练轮数。")
    return parser.parse_args()


def select_experiments(names: list[str] | None) -> list[Experiment]:
    """按名称筛选实验。"""
    experiments = list(SCREEN_EXPERIMENTS)
    if not names:
        return experiments

    by_name = {exp.name: exp for exp in experiments}
    missing = [name for name in names if name not in by_name]
    if missing:
        valid = ", ".join(by_name)
        raise ValueError(f"未知实验名: {', '.join(missing)}。可选实验: {valid}")
    return [by_name[name] for name in names]


def warn_if_data_path_needs_check(data: str) -> None:
    """提示检查旧 Linux 数据路径。"""
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
    """执行筛选或正式实验。"""
    if args.stage == "final" and not args.experiments:
        raise ValueError("final 阶段必须用 --experiments 指定 1-2 个筛选胜出的实验。")

    final = args.stage == "final"
    project = Path(args.project).resolve()
    experiments = select_experiments(args.experiments)
    warn_if_data_path_needs_check(args.data)

    for index, exp in enumerate(experiments, start=1):
        if isinstance(exp.model, Path) and not exp.model.exists():
            raise FileNotFoundError(f"{exp.name} 需要权重文件，但不存在: {exp.model}")

        kwargs = exp.train_kwargs(args, project, final=final)
        print(f"\n{'=' * 80}")
        print(f"实验 {index}/{len(experiments)}: {kwargs['name']}")
        print(f"说明: {exp.desc}")
        print(f"模型: {exp.model}")
        print(f"输出: {project / kwargs['name']}")
        print(f"关键参数: epochs={kwargs['epochs']}, imgsz={kwargs['imgsz']}, batch={kwargs['batch']}, "
              f"degrees={kwargs['degrees']}, mosaic={kwargs['mosaic']}, sndq={kwargs['sndq']}, "
              f"sndq_gamma={kwargs['sndq_gamma']}, sndq_margin={kwargs['sndq_margin']}")
        print(f"{'=' * 80}\n")

        if args.dry_run:
            continue

        model = YOLO(str(exp.model))
        model.train(**kwargs)
        torch.cuda.empty_cache()


def best_metrics(results_csv: Path) -> dict[str, float] | None:
    """读取单个 results.csv 的最佳指标。"""
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
    best_map50_row = max(rows, key=lambda row: get(row, map50_key))
    best_map95_row = max(rows, key=lambda row: get(row, map95_key))
    return {
        "epochs": float(len(rows)),
        "best_epoch_mAP50": get(best_map50_row, "epoch"),
        "best_mAP50": get(best_map50_row, map50_key),
        "best_epoch_mAP50_95": get(best_map95_row, "epoch"),
        "best_mAP50_95": get(best_map95_row, map95_key),
        "precision_at_best_mAP95": get(best_map95_row, precision_key),
        "recall_at_best_mAP95": get(best_map95_row, recall_key),
    }


def summarize(project: Path) -> None:
    """汇总筛选结果，并标记是否达到进入正式训练的门槛。"""
    print("\n第二轮筛选标准:")
    print(f"- mAP50:95 >= {SCREEN_PASS_MAP50_95:.5f} 或 mAP50 >= {SCREEN_PASS_MAP50:.5f}")
    print(f"- 且 Precision >= {SCREEN_MIN_PRECISION:.5f}")
    print("\n实验汇总:")
    print("name,best_mAP50,best_mAP50_95,precision_at_best_mAP95,recall_at_best_mAP95,pass")

    for exp in SCREEN_EXPERIMENTS:
        metrics = best_metrics(project / exp.name / "results.csv")
        if metrics is None:
            print(f"{exp.name},未完成,未完成,未完成,未完成,NO")
            continue

        pass_map = metrics["best_mAP50_95"] >= SCREEN_PASS_MAP50_95 or metrics["best_mAP50"] >= SCREEN_PASS_MAP50
        pass_precision = metrics["precision_at_best_mAP95"] >= SCREEN_MIN_PRECISION
        passed = "YES" if pass_map and pass_precision else "NO"
        print(
            f"{exp.name},"
            f"{metrics['best_mAP50']:.5f},"
            f"{metrics['best_mAP50_95']:.5f},"
            f"{metrics['precision_at_best_mAP95']:.5f},"
            f"{metrics['recall_at_best_mAP95']:.5f},"
            f"{passed}"
        )


def main() -> None:
    """入口函数。"""
    args = parse_args()
    project = Path(args.project).resolve()
    if args.stage == "summary":
        summarize(project)
        return

    run_experiments(args)
    if not args.dry_run:
        summarize(project)


if __name__ == "__main__":
    main()
