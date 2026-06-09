"""codex4 严谨模块改进与轻量化正式实验脚本。

默认每个候选训练 300 epoch，输出到 runs/codex4。
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import torch  # type: ignore
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = "EVD4UAV.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "codex4"
CODEX2_BEST_MAP50_95 = 0.50342
LIGHT_MAP50_95_FLOOR = 0.49600


@dataclass(frozen=True)
class Experiment:
    """单个 codex4 正式候选。"""

    name: str
    model: str
    desc: str
    kind: str
    epochs: int = 300
    imgsz: int = 640
    batch: int = 64
    extra: dict[str, Any] = field(default_factory=dict)

    def train_kwargs(self, args: argparse.Namespace, project: Path, finetune: bool = False) -> dict[str, Any]:
        """生成 Ultralytics 训练参数。"""
        name = f"FinalFT_{self.name}" if finetune else self.name
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
            save_period=-1,
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
            sndq_mode="replace",
            epochs=self.epochs,
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
        if finetune:
            kwargs.update(
                epochs=args.finetune_epochs,
                lr0=0.0003,
                lrf=0.05,
                warmup_epochs=2,
                mosaic=0.0,
                close_mosaic=0,
            )
        return kwargs


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        name="Codex4_Control_Exp04",
        model="ultralytics/cfg/models/11/yolo11-codex4-control-exp04.yaml",
        desc="原 exp4 结构，作为 300 epoch 同环境 control。",
        kind="map",
    ),
    Experiment(
        name="Codex4_SCSPv2",
        model="ultralytics/cfg/models/11/yolo11-codex4-scspv2.yaml",
        desc="只替换 SCSPDRCFv2，验证有界轻量细节回放能否提升 mAP。",
        kind="map",
    ),
    Experiment(
        name="Codex4_NoDy_P3P4Lite",
        model="ultralytics/cfg/models/11/yolo11-codex4-nody-p3p4lite.yaml",
        desc="取消 DySample，P3/P4 使用 C3k2Lite，主看实际 FPS。",
        kind="light",
    ),
    Experiment(
        name="Codex4_SlimHead",
        model="ultralytics/cfg/models/11/yolo11-codex4-slimhead.yaml",
        desc="取消 DySample 并压缩 P4/P5 head 通道，主看 Params/GFLOPs。",
        kind="light",
        batch=64,
    ),
    Experiment(
        name="Codex4_SNDQv2_Aux",
        model="ultralytics/cfg/models/11/yolo11-codex4-sndqv2-aux.yaml",
        desc="SNDQ v2 弱辅助损失，CIoU 仍为主框损失。",
        kind="map",
        extra=dict(sndq=True, sndq_mode="aux", sndq_gamma=0.008, sndq_kappa=32.0, sndq_margin=0.05),
    ),
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 codex4 300 epoch 正式消融实验。")
    parser.add_argument("--stage", choices=("train", "profile", "summary", "final-finetune"), default="train")
    parser.add_argument("--experiments", nargs="*", help="只运行或汇总指定实验名。")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="输出目录，默认 runs/codex4。")
    parser.add_argument("--data", default=DEFAULT_DATA, help="数据集 YAML。")
    parser.add_argument("--device", default="0", help="训练或测速设备。")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers。")
    parser.add_argument("--cache", action="store_true", help="启用 Ultralytics 数据缓存。")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名实验目录。")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要运行的实验。")
    parser.add_argument("--finetune-epochs", type=int, default=80, help="final-finetune 阶段轮数。")
    parser.add_argument("--latency-iters", type=int, default=30, help="profile/summary 中的测速迭代数，0 表示跳过。")
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


def resolve_device(device: str) -> torch.device:
    """将命令行设备字符串转换为 torch.device。"""
    if device != "cpu" and torch.cuda.is_available():
        return torch.device(f"cuda:{str(device).split(',')[0]}")
    return torch.device("cpu")


def profile_model(model_path: str, imgsz: int, device: str, latency_iters: int) -> dict[str, float | str]:
    """统计模型参数、GFLOPs 和可选随机输入延迟。"""
    yolo = YOLO(model_path)
    model = yolo.model
    params = sum(p.numel() for p in model.parameters())
    gflops = float(get_flops(model, imgsz))
    result: dict[str, float | str] = {"params": float(params), "gflops": gflops, "latency_ms": "NA", "fps": "NA"}
    if latency_iters <= 0:
        return result

    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    x = torch.zeros(1, 3, imgsz, imgsz, device=torch_device)
    warmup = max(3, latency_iters // 5)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        start = time.perf_counter()
        for _ in range(latency_iters):
            _ = model(x)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
    latency = (time.perf_counter() - start) * 1000 / latency_iters
    result["latency_ms"] = latency
    result["fps"] = 1000 / latency if latency > 0 else 0.0
    return result


def run_experiments(args: argparse.Namespace, finetune: bool = False) -> None:
    """执行训练或最终短微调。"""
    project = Path(args.project).resolve()
    experiments = select_experiments(args.experiments)
    warn_if_data_path_needs_check(args.data)

    for index, exp in enumerate(experiments, start=1):
        model_path = exp.model
        if finetune:
            weight = project / exp.name / "weights" / "best.pt"
            if not weight.exists():
                raise FileNotFoundError(f"未找到 {exp.name} 的 best.pt，无法 final-finetune: {weight}")
            model_path = str(weight)
        kwargs = exp.train_kwargs(args, project, finetune=finetune)
        print(f"\n{'=' * 80}")
        print(f"实验 {index}/{len(experiments)}: {kwargs['name']}")
        print(f"说明: {exp.desc}")
        print(f"模型: {model_path}")
        print(f"输出: {project / kwargs['name']}")
        print(
            f"关键参数: epochs={kwargs['epochs']}, imgsz={kwargs['imgsz']}, batch={kwargs['batch']}, "
            f"mosaic={kwargs['mosaic']}, sndq={kwargs['sndq']}, sndq_mode={kwargs['sndq_mode']}"
        )
        print(f"{'=' * 80}\n")
        if args.dry_run:
            continue
        model = YOLO(model_path)
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


def summarize(args: argparse.Namespace) -> None:
    """汇总训练指标和结构指标。"""
    project = Path(args.project).resolve()
    experiments = select_experiments(args.experiments)
    print("\ncodex4 判定标准:")
    print(f"- mAP 候选: mAP50:95 > {CODEX2_BEST_MAP50_95:.5f}")
    print(f"- 轻量候选: mAP50:95 >= {LIGHT_MAP50_95_FLOOR:.5f} 且 Params/GFLOPs/Latency 有明显收益")
    print("\n实验汇总:")
    print("name,kind,best_mAP50,best_mAP50_95,precision,recall,params,gflops,size_MB,latency_ms,FPS,tag")

    for exp in experiments:
        metrics = best_metrics(project / exp.name / "results.csv")
        profile = profile_model(exp.model, exp.imgsz, args.device, args.latency_iters)
        weight = project / exp.name / "weights" / "best.pt"
        size_mb = weight.stat().st_size / 1024 / 1024 if weight.exists() else 0.0
        if metrics is None:
            print(
                f"{exp.name},{exp.kind},未完成,未完成,未完成,未完成,"
                f"{profile['params']:.0f},{profile['gflops']:.3f},{size_mb:.2f},"
                f"{profile['latency_ms']},{profile['fps']},NO"
            )
            continue

        if exp.kind == "map" and metrics["best_mAP50_95"] > CODEX2_BEST_MAP50_95:
            tag = "MAP"
        elif exp.kind == "light" and metrics["best_mAP50_95"] >= LIGHT_MAP50_95_FLOOR:
            tag = "LIGHT"
        else:
            tag = "NO"
        latency = profile["latency_ms"]
        fps = profile["fps"]
        latency_str = f"{latency:.2f}" if isinstance(latency, float) else str(latency)
        fps_str = f"{fps:.1f}" if isinstance(fps, float) else str(fps)
        print(
            f"{exp.name},{exp.kind},"
            f"{metrics['best_mAP50']:.5f},{metrics['best_mAP50_95']:.5f},"
            f"{metrics['precision_at_best_mAP95']:.5f},{metrics['recall_at_best_mAP95']:.5f},"
            f"{profile['params']:.0f},{profile['gflops']:.3f},{size_mb:.2f},"
            f"{latency_str},{fps_str},{tag}"
        )


def profile_experiments(args: argparse.Namespace) -> None:
    """打印候选模型结构和延迟。"""
    for exp in select_experiments(args.experiments):
        profile = profile_model(exp.model, exp.imgsz, args.device, args.latency_iters)
        print(
            f"{exp.name}: params={profile['params']:.0f}, gflops={profile['gflops']:.3f}, "
            f"latency_ms={profile['latency_ms']}, fps={profile['fps']}"
        )


def main() -> None:
    """入口函数。"""
    args = parse_args()
    if args.stage == "profile":
        profile_experiments(args)
    elif args.stage == "summary":
        summarize(args)
    elif args.stage == "final-finetune":
        run_experiments(args, finetune=True)
    else:
        run_experiments(args)


if __name__ == "__main__":
    main()
