# 处理后数据目录

此目录用于保存 ROI 裁剪、肺野预处理或其他数据处理后的结果。

当前项目主线使用原始胸片目录直接训练；HybridGNet ROI 预处理属于历史/备选方案，相关脚本位于 `data/datasets/precompute_hybridgnet_roi.py`。

经过测试后我们组放弃了HybridGNet ROI预处理。