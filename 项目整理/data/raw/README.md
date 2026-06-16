# 原始数据目录

此目录用于放置原始 Chest X-Ray Images (Pneumonia) 数据。

提交代码包默认不包含大体积原始图片。运行训练前，请将数据整理为：

```text
data/raw/chest_xray/
  train/NORMAL/
  train/PNEUMONIA/
  val/NORMAL/
  val/PNEUMONIA/
  test/NORMAL/
  test/PNEUMONIA/
```

原始工程中使用的数据路径为 `../dataset/chest_xray`。

