# rasterizer

扫描线多边形光栅化内核：顶点用定点整数表示，一个像素分成 SUBPIXELS 个子像素步长，边表建好后逐行推进
活性边表，按 even-odd 或 nonzero 填充规则配对交点，输出被覆盖的像素集合，可再按像素矩形裁剪，纯标准库 Python 3。

## 内容

- rasterizer/core.py：ScanlineRasterizer、Edge、RasterizerError、SUBPIXELS、FILL_RULES
- tests/test_core.py：unittest 用例

像素 (column, row) 的采样点落在该像素中心；边界按半开处理，压在跨度右边或下边的采样点算在轮廓外。

## 跑测试

在项目根目录执行：

    python3 -m unittest discover -s tests -v
