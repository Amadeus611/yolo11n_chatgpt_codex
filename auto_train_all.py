import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".mplconfig"))
import torch  # type: ignore
from ultralytics import YOLO


def main():
    # =========================================================
    # SCSP + DRCF + SNDQ + LiteNeck 完整消融实验任务列表
    # ---------------------------------------------------------
    # Table 1: 主创新 SCSP + DRCF 消融 (Exp01-03)
    # Table 2: 保守 LiteNeck 消融 (Exp04)
    # Table 3: 全部启用 (Exp05)
    # Table 4: 强基线对比 (Exp06)
    # =========================================================
    experiments = [
        # =====================================================
        # Table 1: SCSP + DRCF 主创新消融实验
        # =====================================================
        {
            "yaml": "ultralytics/cfg/models/11/yolo11n.yaml",
            "name": "Exp01_Baseline",
            "sndq": False,
        },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11n-scsp-drcf.yaml",
            "name": "Exp02_SCSP_DRCF",
            "sndq": False,
        },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11n-scsp-drcf.yaml",
        #     "name": "Exp03_SCSP_DRCF_SNDQ",
        #     "sndq": True,
        # },

        # =====================================================
        # Table 2: LiteNeck 速度消融
        # =====================================================
        {
            "yaml": "ultralytics/cfg/models/11/yolo11n-scsp-drcf-litenneck.yaml",
            "name": "Exp04_SCSP_DRCF_LiteNeck",
            "sndq": False,
        },

        # =====================================================
        # Table 3: 全部创新启用
        # =====================================================
        {
            "yaml": "ultralytics/cfg/models/11/yolo11n-scsp-drcf-sndq-litenneck.yaml",
            "name": "Exp05_Full_SCSP_DRCF_SNDQ_LiteNeck",
            "sndq": True,
        },

        # =====================================================
        # Table 4: 强基线对比
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26n.yaml",
        #     "name": "Exp06_YOLO26n",
        #     "sndq": False,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/rt-detr/rtdetr-l.yaml",
        #     "name": "Exp07_RTDETR_l",
        #     "sndq": False,
        # },
    ]

    # =========================================================
    # 循环执行实验
    # =========================================================
    for i, exp in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"  实验 {i + 1}/{len(experiments)}: {exp['name']}")
        print(f"  配置文件: {exp['yaml']}")
        print(f"  SNDQ Loss: {exp['sndq']}")
        print(f"{'=' * 60}\n")

        model = YOLO(exp["yaml"])

        common_kwargs = dict(
            # --- 数据集 ---
            data="EVD4UAV.yaml",
            imgsz=640,
            batch=64,
            name=exp["name"],
            project="/home/ssssss/1yolo/Ablation_Results",
            device=0,
            workers=8,
            val=True,
            plots=True,
            save=True,
            amp=True,
            deterministic=False,  # DySample 使用 grid_sample，关闭强确定性可避免 CUDA 反向警告
            cache=False,

            # --- 优化器 ---
            optimizer="SGD",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            cos_lr=True,
            warmup_epochs=5,

            # --- 损失权重 ---
            box=exp.get("box", 7.5),
            cls=exp.get("cls", 0.5),
            dfl=exp.get("dfl", 1.5),
            cls_pw=exp.get("cls_pw", 0.3),  # 类别平衡: 温和逆频率加权，实际权重由训练集类别频次自动计算

            # --- SNDQ ---
            sndq=exp["sndq"],
            sndq_gamma=exp.get("sndq_gamma", 0.05),
            sndq_tau=exp.get("sndq_tau", 32.0),
            sndq_c=exp.get("sndq_c", 12.8),
            sndq_kappa=exp.get("sndq_kappa", 64.0),
            sndq_margin=exp.get("sndq_margin", 0.0),

            # --- 训练策略 ---
            epochs=300,
            patience=100,

            # --- 数据增强 (航拍适配) ---
            mosaic=0.8,
            close_mosaic=40,
            mixup=0.0,
            copy_paste=0.1,  # HBB 检测中 CopyPaste 不宜过强，避免制造不自然遮挡
            degrees=25.0,  # 航拍车辆方向多变，但过大旋转会让 HBB 外接框膨胀
            scale=0.25,
            translate=0.08,
            fliplr=0.5,
            flipud=0.5,  # 俯视航拍上下翻转通常合理，可增加方向鲁棒性
            erasing=0.0,  # 检测任务中随机擦除容易误伤小车目标，先关闭
            hsv_h=0.01,
            hsv_s=0.4,
            hsv_v=0.3,

            # --- 正则化 ---
            dropout=0.0,
        )

        # 透传 SNDQ 内部参数
        for k in ("sndq_gamma", "sndq_tau", "sndq_c", "sndq_kappa", "sndq_margin"):
            if k in exp:
                common_kwargs[k] = exp[k]

        model.train(**common_kwargs)
        torch.cuda.empty_cache()

    print("\n  所有消融实验已全部执行完毕！")


if __name__ == "__main__":
    main()
