# 项目上下文速览

本文档用于在新对话中快速了解当前项目状态，重点记录已经完成的 YOLO11n 改进、训练脚本约定、测试结果和后续注意事项。

## 项目基本信息

- 论文题目：《基于深度学习的无人机航拍车辆识别方法设计》
- 基础框架：Ultralytics YOLO11n
- 研究方向：无人机航拍车辆水平框检测，当前不把 OBB 作为主线
- 目标数据集：EVD4UAV，类别为 `car`、`bus`、`truck`
- 当前主线方案：`YOLO11n + SCSP + DRCF + SNDQ + 保守 LiteNeck`
- 暂缓模块：NCDH，不作为主线实现，只建议后续作为失败/增强消融尝试

## 本地环境

- 系统：Windows 11 x64
- 默认 Python 环境：`D:\miniconda\envs\yolo11\python.exe`
- 运行 Python 前建议设置：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH='D:\1yolo\yolo11n_chatgpt_codex'
$env:MPLCONFIGDIR='D:\1yolo\yolo11n_chatgpt_codex\.mplconfig'
```

正式训练可执行：

```powershell
D:\miniconda\envs\yolo11\python.exe auto_train_all.py
```

## 数据集配置

数据配置文件为 `EVD4UAV.yaml`。

当前需要特别注意：`path` 仍然是旧 Linux 路径：

```yaml
path: /home/ssssss/1yolo/Dataset/EVD4UAV
```

正式训练前必须改成当前 Windows 机器上的真实 EVD4UAV 数据集根目录。本轮没有擅自修改该路径，因为本机没有找到 EVD4UAV 数据目录。

## 已实现模块

### SCSP

位置：`ultralytics/nn/modules/block.py`

用途：用高层语义特征对浅层细节特征做弱门控净化，减少道路纹理、停车线、树影等浅层噪声反灌 Neck。

实现原则：

- 输入浅层特征 `F_s` 和高层语义特征 `F_h`
- 输出净化后的 detail feature
- 通道数保持较轻
- 门控是弱约束，不做强抑制

### DRCF

位置：`ultralytics/nn/modules/block.py`

用途：把 SCSP 净化后的浅层细节以残差方式注入 P3，不新增 P2 检测头。

推荐形式已经按计划实现为：

```text
P3_out = P3 + alpha * detail
```

其中 `alpha` 为可学习参数，初始化为小值，避免训练初期扰乱原 YOLO11n 表征。

### SNDQ

位置：`ultralytics/utils/loss.py`

用途：替换默认 box loss 中的部分 IoU 表达，增强小目标和密集邻近车辆场景下的框质量约束。

实现范围：

- 只改 box loss
- 不改 DFL
- 不改 TAL
- 不改 assigner
- 支持无邻居时退化为普通 IoU/NWD 混合损失

关键参数来自模型训练参数或 YAML：

- `sndq`
- `sndq_gamma`
- `sndq_tau`
- `sndq_c`
- `sndq_kappa`
- `sndq_margin`

注意事项：

- `sndq_gamma` 必须保守，建议 `0.05` 或 `0.1`
- 邻居惩罚过强可能误伤密集停车场中的合理框，导致 Precision 下降
- 曾修复过一个反向传播问题：SNDQ 内部不能使用原地 `clamp_()`，否则会破坏 autograd；现在使用非原地 `clamp()`

### LiteNeck

位置：`ultralytics/nn/modules/block.py`

新增模块：

- `PartialConv`
- `PConvBottleneck`
- `C3k2Lite`
- `DySample`

实现原则：

- 只做保守替换
- 只替换一个 P3 高分辨率融合块为 `C3k2Lite`
- 只把一条 `P4 -> P3` 上采样路径换成 `DySample`
- 目标主要是抵消 SCSP/DRCF/SNDQ 带来的速度损失，不保证单独提升精度

注意事项：

- `DySample` 使用 `grid_sample`
- CUDA backward 不是强确定性实现
- 训练脚本中已设置 `deterministic=False`，避免相关警告/报错干扰训练

## 关键改动文件

- `auto_train_all.py`
  - 主训练脚本，保留旧脚本大体结构
  - 当前按消融顺序组织实验
  - 已加入 SNDQ 参数、增强参数、损失权重、`deterministic=False`

- `EVD4UAV.yaml`
  - 数据集配置
  - 正式训练前重点检查 `path`

- `ultralytics/nn/modules/block.py`
  - 新增 SCSP、DRCF、LiteNeck 相关模块

- `ultralytics/nn/modules/__init__.py`
  - 导出新增模块

- `ultralytics/nn/tasks.py`
  - 注册新增模块，使 YAML 能正常解析

- `ultralytics/utils/loss.py`
  - 实现 SNDQ box loss

- `ultralytics/cfg/default.yaml`
  - 添加 SNDQ 默认配置项

- `ultralytics/cfg/models/11/yolo11-scsp-drcf.yaml`
  - `SCSP + DRCF` 结构

- `ultralytics/cfg/models/11/yolo11-scsp-drcf-litenneck.yaml`
  - `SCSP + DRCF + LiteNeck` 结构

- `ultralytics/cfg/models/11/yolo11-scsp-drcf-sndq-litenneck.yaml`
  - `SCSP + DRCF + SNDQ + LiteNeck` 完整结构

## 当前训练脚本实验顺序

`auto_train_all.py` 中当前实验大致为：

1. `Exp01_Baseline`
2. `Exp02_SCSP_DRCF`
3. `Exp03_SCSP_DRCF_SNDQ`
4. `Exp04_SCSP_DRCF_LiteNeck`
5. `Exp05_Full_SCSP_DRCF_SNDQ_LiteNeck`
6. `Exp06_YOLO26n`

建议论文消融重点仍按以下顺序分析：

1. YOLO11n
2. YOLO11n + SCSP
3. YOLO11n + SCSP + DRCF
4. YOLO11n + SCSP + DRCF + SNDQ
5. YOLO11n + SCSP + DRCF + SNDQ + LiteNeck
6. 可选 NCDH

当前代码里 SCSP 和 DRCF 是组合结构，若后续需要严格拆分 `SCSP` 与 `SCSP + DRCF` 两组，需要再新增一个只含 SCSP 的 YAML。

## 当前增强参数判断

当前脚本里的增强参数是偏保守、适合无人机航拍车辆的设置：

- `mosaic=0.8`
- `close_mosaic=40`
- `mixup=0.0`
- `copy_paste=0.1`
- `degrees=20.0`
- `scale=0.25`
- `translate=0.08`
- `fliplr=0.5`
- `flipud=0.5`
- `erasing=0.0`
- `hsv_h=0.01`
- `hsv_s=0.4`
- `hsv_v=0.3`

主要考虑：

- 航拍车辆方向变化明显，`flipud=0.5` 合理
- 小目标检测不宜使用过强 `scale` 和 `erasing`
- `mixup=0.0` 避免密集车辆边界被混合污染
- `copy_paste=0.1` 可以小幅增加目标组合，但不宜太强
- `close_mosaic=40` 用于训练后期回归真实图像分布

## 当前损失权重判断

当前脚本中损失权重为：

- `box=7.5`
- `cls=0.5`
- `dfl=1.5`
- `cls_pw=0.3`

判断：

- `box=7.5` 和 `dfl=1.5` 适合检测任务主线，不建议一开始改
- `cls=0.5` 对三类车辆任务较稳
- `cls_pw=0.3` 是温和类别平衡，实际权重由训练集类别频次自动计算
- 若 SNDQ 导致 Precision 明显下降，优先减小 `sndq_gamma`，不要先动 `box/dfl`

## 已完成运行测试

已用本地 `yolo11` 虚拟环境测试。

通过项：

- `auto_train_all.py` 语法检查通过
- 改进模型 YAML 均可正常加载
- 模型 stride 正常为 `[8, 16, 32]`
- SNDQ 最小前向/反向测试通过
- CPU 上完整模型 1 epoch 冒烟训练通过
- GPU 上完整模型 1 epoch 冒烟训练通过
- GPU 上 baseline YOLO11n 1 epoch 冒烟训练通过

本机 GPU：

```text
NVIDIA GeForce RTX 3060 Laptop GPU, 6144MiB
```

## 后续实验判断标准

建议每组至少记录：

- `mAP50`
- `mAP50:95`
- `AP_small`
- `Precision`
- `Recall`
- `Params`
- `FLOPs`
- `FPS / latency`

判断建议：

- SCSP + DRCF 如果不提升 `AP_small` 或 `Recall`，不要急着继续叠复杂模块
- SNDQ 如果只涨 `AP_small` 但明显掉 `Precision`，减小 `sndq_gamma`
- LiteNeck 如果掉点超过 `0.3 mAP50:95`，只作为速度实验，不进最终主模型
- NCDH 如果普通稀疏场景 Precision 下降，直接作为失败消融，不强行保留

## 新对话接手建议

新对话中优先检查：

1. `EVD4UAV.yaml` 的数据集路径是否已经改成 Windows 本机真实路径
2. `auto_train_all.py` 中实验列表是否符合当前要跑的消融顺序
3. 是否需要补一个只含 SCSP 的 YAML，用于更严格拆分消融
4. 正式训练前先小跑 1 epoch，确认数据集路径、类别数、缓存和显存都正常

如果目标是继续写代码，优先不要重写整体工程；沿着现有 Ultralytics 模块注册方式和 YAML 结构做小范围增量修改。
