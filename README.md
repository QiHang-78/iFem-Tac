# iFEM-Tac

实时读取相机画面，基于标定参数和 25 个 marker 参考点生成去 marker 的深度图，并保留半球挖除部分的重建数据。

## 目录

```text
config/          标定文件
data/source/     原始 OBJ 数据
data/reference/  参考图、marker 三维点、重建后的半球挖除模型
docs/            marker 处理说明
scripts/         可运行脚本
test_logs/       测试截图、深度图和统计输出，已加入 .gitignore
```

## 常用命令

实时显示三列窗口：

```powershell
conda run --no-capture-output -n gelsight python scripts/live_marker_removed_depth.py
```

保存一帧测试输出到 `test_logs/`：

```powershell
conda run --no-capture-output -n gelsight python scripts/live_marker_removed_depth.py --once --no-window
```

从参考图重新计算 marker 三维参考点：

```powershell
conda run --no-capture-output -n gelsight python scripts/project_markers_to_hemisphere.py
```

从原始 OBJ 重建半球挖除部分：

```powershell
conda run --no-capture-output -n gelsight python scripts/reconstruct_hemisphere_cutout.py
```

## 实时窗口快捷键

`r` 重置当前帧为基准，`i` 显示或隐藏 marker 编号，`s` 保存当前帧输出，`q` 或 `Esc` 退出。
