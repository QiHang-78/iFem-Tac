# Marker 处理说明

本文档整理当前工程中 marker 的处理逻辑。代码里有两类 marker：

- GelSight 表面点阵 marker：用于检测点阵、去除图像中的点、匹配跨帧位移，并生成 `Cx/Cy/Cz`、`Ox/Oy/Oz`、`markerstatus` 等数据。
- ArUco marker：只在 `ForceCalibrationSensor3_slip.py` 中用于外部姿态估计，和 GelSight 表面点阵 marker 是独立流程。

## 相关文件

| 文件 | 作用 |
| --- | --- |
| `ForceCalibrationSensor7_Noslip.py` | 当前 7x7 点阵主流程，包含 marker 检测、去除、匹配、状态保存 |
| `ForceCalibrationSensor3_Noslip_backup.py` | 旧 13x13 点阵流程，整体逻辑相同，阈值和点阵尺寸不同 |
| `ForceCalibrationSensor3_slip.py` | slip 版本，包含点阵 marker 处理和 ArUco 姿态估计 |
| `MultiCam_Calibration.py` | 标定数据预处理，使用 marker mask 修补图像，并在接触检测中排除 marker 区域 |
| `Aberration_cali.py` | 畸变/位置相关标定中的 marker 检测辅助逻辑 |
| `setting_GelForce.py` | 点阵数量、初始位置、间距和匹配帧率配置 |
| `lib/find_marker.pyd` / `lib/find_marker.so` | 点阵 marker 的跨帧匹配实现，Python 侧只能调用接口，内部算法不在源码中 |

## 点阵 marker 主流程

主流程以 `ForceCalibrationSensor7_Noslip.py` 为准：

```text
相机帧
  -> crop_Gel() 裁剪 GelSight 区域
  -> creat_mask_2() 检测 marker 二值图
  -> find_dots() 提取 marker 中心 keypoints
  -> make_mask() 生成修补用 mask
  -> cv2.inpaint() 去除 marker，得到 inpaint_image
  -> preprocessV2() / getdepth() 重建深度图
  -> find_marker.Matching 匹配点阵位移
  -> marker_depthV2() 给 marker 采样 z 值
  -> dispOpticalFlow_new() 判断 markerstatus 并绘制位移
  -> 可选保存 depth/raw/marker mat 数据
```

## 1. marker 二值检测：`creat_mask_2`

位置：

- `ForceCalibrationSensor7_Noslip.py:233`
- `ForceCalibrationSensor3_Noslip_backup.py:215`
- `ForceCalibrationSensor3_slip.py:362`
- `MultiCam_Calibration.py:702`

处理步骤：

1. 对输入图像 `raw_image` 做 `cv2.pyrDown()` 降采样。
2. 做两种尺度的高斯模糊：
   - 大核：例如 `(9, 9)`
   - 小核：例如 `(3, 3)`
3. 用 `diff = blur - blur2` 提取局部高频差异，再乘以放大系数 `15.0`。
4. 将 `diff` 限制到 `[0, 255]`。
5. 对 B/G/R 三个通道分别做阈值判断。
6. 任意两个通道同时超过阈值时，认为该像素属于 marker：

```python
mask = ((mask_b * mask_g) + (mask_b * mask_r) + (mask_g * mask_r)) > 0
```

7. 将 mask resize 回原图尺寸，乘以 `self.dmask` 去掉边界无效区域。
8. 用形态学操作清理 mask。
9. 返回反相图：

```python
return (1 - mask) * 255
```

返回结果中，marker 区域是黑色，背景是白色，便于后续 `SimpleBlobDetector` 找暗色 blob。

不同脚本中的阈值不同：

| 文件 | B/G/R 阈值 | 形态学处理 |
| --- | ---: | --- |
| `ForceCalibrationSensor7_Noslip.py` | `120` | `erode` 后 `dilate` |
| `ForceCalibrationSensor3_Noslip_backup.py` | `210` | `erode` 后 `dilate` |
| `ForceCalibrationSensor3_slip.py` | `60` | `dilate` |
| `MultiCam_Calibration.py` | `150` | `dilate` |

这些阈值和光照、marker 颜色、相机曝光强相关。如果 marker 数量检测不稳定，通常优先调这里。

## 2. marker 中心提取：`find_dots`

位置：

- `ForceCalibrationSensor7_Noslip.py:273`
- `ForceCalibrationSensor3_Noslip_backup.py:255`
- `ForceCalibrationSensor3_slip.py:401`
- `MultiCam_Calibration.py:743`

使用 OpenCV 的 `cv2.SimpleBlobDetector` 从二值图中找点。主要参数：

```python
params.minThreshold = 1
params.maxThreshold = 12
params.minDistBetweenBlobs = 9
params.filterByArea = True
params.minArea = 9
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.minInertiaRatio = 0.5
```

输出是 OpenCV keypoints。每个 keypoint 的中心坐标通过：

```python
keypoints[i].pt[0]  # x
keypoints[i].pt[1]  # y
```

读取。

## 3. marker mask 与图像修补：`make_mask` + `cv2.inpaint`

位置：

- `ForceCalibrationSensor7_Noslip.py:291`
- `ForceCalibrationSensor3_Noslip_backup.py:273`
- `ForceCalibrationSensor3_slip.py:419`
- `MultiCam_Calibration.py:761`

`make_mask()` 会在每个 marker 中心画一个白色椭圆，生成修补 mask：

```python
cv2.ellipse(
    img,
    (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])),
    (5, 5),
    0,
    0,
    360,
    255,
    -1,
)
```

不同脚本的椭圆大小略有区别：

- `ForceCalibrationSensor7_Noslip.py`：`(5, 5)`
- `ForceCalibrationSensor3_Noslip_backup.py`：`(7, 5)`
- `ForceCalibrationSensor3_slip.py`：`(3, 3)`
- `MultiCam_Calibration.py`：`(5, 5)`

生成 mask 后，用 OpenCV 的 Telea inpaint 算法修补 marker 区域：

```python
inpaint_image = cv2.inpaint(raw_image, marker_mask_new, 3, cv2.INPAINT_TELEA)
```

深度重建使用的是 `inpaint_image`，不是原始 `raw_image`。这样可以避免黑色 marker 点影响颜色差分和深度估计。

## 4. 点阵排序：`sortkeypoints`

位置：

- `ForceCalibrationSensor7_Noslip.py:301`
- `ForceCalibrationSensor3_Noslip_backup.py:283`
- `ForceCalibrationSensor3_slip.py:429`
- `MultiCam_Calibration.py:771`

排序逻辑：

```python
xy = sorted(xy, key=lambda x: [x[1], x[0]])
```

即先按 `y` 从上到下，再按 `x` 从左到右排序。这样一帧中的点阵顺序可以和二维数组的行列关系对应。

注意：当前主流程中，点阵匹配实际交给 `find_marker.Matching`。排序更多用于调试、旧 slip 位移计算和部分辅助逻辑。

## 5. 点阵匹配：`find_marker.Matching`

初始化位置：

- `ForceCalibrationSensor7_Noslip.py:98`
- `ForceCalibrationSensor3_Noslip_backup.py:83`
- `ForceCalibrationSensor3_slip.py:93`

匹配器配置来自 `setting_GelForce.py`：

```python
N_ = 7
M_ = 7
fps_ = 30
x0_ = 35
y0_ = 35
dx_ = 45
dy_ = 45
```

含义：

- `N_` / `M_`：marker 点阵行数和列数。
- `x0_` / `y0_`：左上角 marker 的初始坐标。
- `dx_` / `dy_`：相邻 marker 的横向、纵向间距。
- `fps_`：匹配算法期望帧率。

每帧匹配调用：

```python
self.m.init(keypoints_new)
self.m.run()
flow = self.m.get_flow()
```

`flow` 解包为：

```python
Ox, Oy, Cx, Cy, Occupied = flow
```

含义：

- `Ox`, `Oy`：原始/参考点阵坐标。
- `Cx`, `Cy`：当前帧匹配后的点阵坐标。
- `Occupied`：匹配状态，具体语义由 `lib/find_marker` 内部实现决定。

因为 `find_marker` 是 `.pyd/.so` 二进制模块，当前仓库无法直接查看匹配算法源码。Python 侧只负责输入 keypoints 和读取 flow。

## 6. marker 的 z 值：`marker_depthV2`

位置：

- `ForceCalibrationSensor7_Noslip.py:391`
- `ForceCalibrationSensor3_Noslip_backup.py:373`
- `ForceCalibrationSensor3_slip.py:535`

函数逻辑：

```python
Ox, Oy, Cx, Cy, Occupied = flow
Cz = np.zeros_like(Ox)
for i in range(len(Ox)):
    for j in range(len(Ox[i])):
        Cz[i][j] = depth[int(Oy[i][j]), int(Ox[i][j])]
```

注意这里的实现是按 `Ox/Oy` 在深度图上采样，然后结果命名为 `Cz`。后续代码会把这个 `Cz` 和当前帧的 `Cx/Cy` 一起保存或显示。也就是说，当前 z 值来自参考/原始网格坐标处的深度采样，而不是 `Cx/Cy` 当前坐标处的采样。

## 7. 参考帧与当前帧坐标

主要变量：

| 变量 | 含义 |
| --- | --- |
| `Cx`, `Cy`, `Cz` | 当前帧 marker 坐标和深度 |
| `Ox`, `Oy`, `Oz` | 用户设置的参考帧 marker 坐标和深度 |
| `OOx`, `OOy`, `OOz` | 初始轮次记录的原始点阵基准 |
| `markerstatus` | 当前帧每个 marker 是否触发 |
| `refmarkerstatus` | 7x7 主流程中，设置参考帧时记录的触发状态 |

在 `ForceCalibrationSensor7_Noslip.py` 中：

- 首轮运行时记录 `OOx/OOy`。
- 鼠标点击 `Raw` 窗口后，`set_ref()` 将 `self.refmarker = True`。
- 下一帧中，如果 `self.refmarker` 为真，代码把当前 `flow[2]/flow[3]` 和 `Cz` 记录到 `Ox/Oy/Oz`，作为新的参考帧。
- 开启 `drawdiff` 后，显示当前点相对 `OOx/OOy` 或参考点的位移。

## 8. markerstatus 的生成

位置：

- `ForceCalibrationSensor7_Noslip.py:513`
- `ForceCalibrationSensor3_Noslip_backup.py:490`

`markerstatus` 每帧先清零：

```python
self.markerstatus = np.zeros((7, 7)).astype(int)
```

在 `dispOpticalFlow_new()` 中，根据 z 值阈值判断该 marker 是否被触发：

```python
indices = np.where(z > self.depththresh)
...
if (i, j) in zip(*indices):
    self.markerstatus[i, j] = 1
```

当前主脚本 `ForceCalibrationSensor7_Noslip.py` 中：

```python
self.depththresh = 0.4
```

旧 13x13 脚本中：

```python
self.depththresh = 0.3
```

因此：

- `markerstatus[i, j] = 1`：该 marker 对应区域深度超过阈值，认为被触发。
- `markerstatus[i, j] = 0`：未触发。

## 9. 数据保存

在 `ForceCalibrationSensor7_Noslip.py` 中，点击 `Depth` 窗口会切换 `save_alldata`。开启后每帧保存：

| 输出 | 内容 |
| --- | --- |
| `Data/depth_mat/*.mat` | 深度矩阵 `depth` |
| `Data/raw/*.jpg` | 绘制后的 raw 图 |
| `Data/Figure/*.jpg` | 当前 GelSight 裁剪图 |
| `Data/depth/*.jpg` | 彩色深度图 |
| `Data/markers/*.mat` | marker 坐标和状态 |

`Data/markers/*.mat` 中包含：

```python
{
    "Ox": self.Ox,
    "Oy": self.Oy,
    "Oz": self.Oz,
    "Cx": self.Cx,
    "Cy": self.Cy,
    "Cz": self.Cz,
    "OOx": self.OOx,
    "OOy": self.OOy,
    "OOz": self.OOz,
    "status": self.markerstatus,
}
```

## 10. 标定流程中的 marker 处理

`MultiCam_Calibration.py` 中的 marker 处理主要用于生成无 marker 的训练/标定图像：

```text
原始图像
  -> creat_mask_2()
  -> find_dots()
  -> make_mask()
  -> cv2.inpaint()
  -> 保存到 TacData_fill/
```

接触区域检测 `contact_detection()` 中也会使用 `marker_mask` 排除 marker 区域：

```python
contact_mask = (diff_img > 18).astype(np.uint8) * (1 - marker_mask)
```

这样标定球接触区域不会被 marker 点误判。

## 11. ArUco marker 流程

ArUco marker 只出现在 `ForceCalibrationSensor3_slip.py` 的 `pose_esitmation()` 中，和表面点阵 marker 不是同一套流程。

处理步骤：

1. 对整帧图像做 `cv2.medianBlur(frame, 3)`。
2. 转灰度。
3. 用 `cv2.aruco.detectMarkers()` 检测 ArUco。
4. 分别寻找 id 为 `1` 和 `2` 的 marker。
5. 用 `cv2.aruco.estimatePoseSingleMarkers()` 估计每个 ArUco 的 `rvec/tvec`。
6. 对 `tvec` 做指数平滑：

```python
self.tvec1 = self.alpha * self.tvec1 + (1 - self.alpha) * self.pre_tvec1
```

7. 将两个 ArUco 的位移合并，并减去 `sensor_bias`。
8. 用 `cv2.aruco.drawDetectedMarkers()` 可视化检测结果。

这里的 ArUco 用于外部位姿或位移估计，不参与 `markerstatus`、`Cx/Cy/Cz` 的点阵输出。

## 12. 调参建议

常见问题和对应参数：

| 现象 | 优先检查 |
| --- | --- |
| marker 检测数量少 | 降低 `creat_mask_2()` 中 B/G/R 阈值 |
| marker 检测出大片噪声 | 提高 B/G/R 阈值，或加强 erode |
| 相邻 marker 粘连 | 增大腐蚀，减小 `make_mask()` 椭圆，检查曝光 |
| blob 数量不稳定 | 调 `minArea`、`minDistBetweenBlobs`、形态学核大小 |
| 深度图受 marker 黑点影响 | 检查 `marker_mask_new` 是否覆盖完整，增大 inpaint 椭圆 |
| `markerstatus` 误触发 | 调整 `self.depththresh` |
| 改成不同点阵尺寸后匹配错乱 | 同步修改 `setting_GelForce.py` 中 `N_/M_/x0_/y0_/dx_/dy_`，并确认数组初始化尺寸一致 |

建议调试时临时打开这些窗口：

```python
cv2.imshow("marker", marker_new)
cv2.imshow("marker_mask", marker_mask_new)
cv2.imshow("inpaint", inpaint_image)
cv2.imshow("Depth", depth2)
```

先确认 marker 检测和修补正确，再看匹配和 `markerstatus`。

## 13. 当前实现需要注意的点

- `ForceCalibrationSensor7_Noslip.py` 是 7x7 点阵；`readme.txt` 和 `ForceCalibrationSensor3_Noslip_backup.py` 仍保留 13x13 描述，属于历史版本差异。
- `find_marker.Matching` 的内部实现不可见，调试匹配问题时只能从输入 keypoints、点阵配置和输出 flow 侧定位。
- `marker_depthV2()` 当前按 `Ox/Oy` 采样深度，但结果命名为 `Cz`。如果后续需要严格表示当前点的 z 值，应评估是否改为按 `Cx/Cy` 采样。
- `creat_mask_2`、`make_mask` 在不同脚本中的阈值和椭圆大小不同，迁移参数时不要只改一个文件。
