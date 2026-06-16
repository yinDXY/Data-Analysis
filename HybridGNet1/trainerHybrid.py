import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import argparse

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import scipy.sparse as sp
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import hausdorff_distance as hd

# 导入自定义模块
from utils.utils import scipy_to_torch_sparse, genMatrixes, genMatrixesLH
from utils.dataLoader import LandmarksDataset, ToTensor, ToTensorLH, Rescale, RandomScale, AugColor, Rotate
from models.hybridDoubleSkip import Hybrid as DoubleSkip
from models.hybridSkip import Hybrid as Skip
from models.hybrid import Hybrid
from models.hybridNoPool import Hybrid as HybridNoPool
from models.chebConv import Pool

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 类型别名
Tensor = torch.Tensor
ConfigType = Dict[str, Union[int, float, bool, str, List[int]]]


def hd_land(target: np.ndarray, pred: np.ndarray, shape: Tuple[int, int]) -> float:
    """
    计算两组地标点之间的豪斯多夫距离
    """
    # 确保坐标在有效范围内
    target = target.clip(0, shape[0] - 1)
    pred = pred.clip(0, shape[0] - 1)
    
    # 创建坐标掩码
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    
    # 填充有效坐标
    valid_target = (target[:, 0] >= 0) & (target[:, 1] >= 0)
    valid_pred = (pred[:, 0] >= 0) & (pred[:, 1] >= 0)
    
    target_valid = target[valid_target]
    pred_valid = pred[valid_pred]
    
    if len(target_valid) == 0 or len(pred_valid) == 0:
        return float("inf")
    
    coords_a[target_valid[:, 0].astype(int), target_valid[:, 1].astype(int)] = True
    coords_b[pred_valid[:, 0].astype(int), pred_valid[:, 1].astype(int)] = True
    
    return hd(coords_a, coords_b)


def hd_landmarks(out: Tensor, label: Tensor, size: int = 512) -> float:
    """
    计算不同器官地标点的平均豪斯多夫距离
    """
    # 转换为numpy数组并缩放
    target = np.round(label.cpu().numpy() * size).astype(np.int32)
    pred = np.round(out.cpu().numpy() * size).astype(np.int32)
    shape = (size, size)
    
    # 计算不同器官的HD
    d_lungs = hd_land(target[:94, :], pred[:94, :], shape)
    d_heart = hd_land(target[94:120, :], pred[94:120, :], shape)
    
    # 根据是否包含锁骨点计算平均
    if target.shape[0] > 120:
        d_cla = hd_land(target[120:, :], pred[120:, :], shape)
        return (d_lungs + d_heart + d_cla) / 3
    else:
        return (d_lungs + d_heart) / 2


def calculate_loss(out: Union[Tensor, Tuple[Tensor, ...]], target: Tensor, 
                  target_down: Tensor) -> Tuple[Tensor, Tensor]:
    """
    统一计算不同模型输出的损失
    """
    if isinstance(out, Tensor):
        # 基础模型：仅最终输出
        outloss = F.mse_loss(out, target)
        loss = outloss
    elif len(out) == 2:
        # 单跳连接模型
        out, pre = out
        preloss = F.mse_loss(pre, target_down)
        outloss = F.mse_loss(out, target)
        loss = outloss + preloss
    elif len(out) == 3:
        # 双跳连接模型
        out, pre1, pre2 = out
        pre1loss = F.mse_loss(pre1, target_down)
        pre2loss = F.mse_loss(pre2, target)
        outloss = F.mse_loss(out, target)
        loss = outloss + pre1loss + pre2loss
    else:
        raise ValueError(f"不支持的输出格式: 长度 {len(out)}")
    
    return loss, outloss


def create_datasets(config: ConfigType) -> Tuple[LandmarksDataset, LandmarksDataset]:
    """
    创建训练和验证数据集
    """
    # 设置数据路径
    if config["extended"]:
        train_path = Path("Datasets/Extended/Train")
        val_path = Path("Datasets/Extended/Val")
    else:
        train_path = Path("Datasets/JSRT/Train")
        val_path = Path("Datasets/JSRT/Val")
    
    # 验证路径存在性
    for path in [train_path, val_path]:
        if not path.exists():
            raise FileNotFoundError(f"数据路径不存在: {path}")
    
    # 选择合适的Tensor转换类
    to_tensor = ToTensor if config["allOrgans"] else ToTensorLH
    input_size = config["inputsize"]
    
    # 训练集变换（包含数据增强）
    train_transform = transforms.Compose([
        RandomScale(),
        Rotate(3),
        AugColor(0.40),
        to_tensor()
    ])
    
    # 验证集变换（仅缩放）
    val_transform = transforms.Compose([
        Rescale(input_size),
        to_tensor()
    ])
    
    # 创建数据集
    train_dataset = LandmarksDataset(
        img_path=str(train_path / "Images"),
        label_path=str(train_path / "landmarks"),
        transform=train_transform
    )
    
    val_dataset = LandmarksDataset(
        img_path=str(val_path / "Images"),
        label_path=str(val_path / "landmarks"),
        transform=val_transform
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_model(config: ConfigType, D_t: List[Tensor], U_t: List[Tensor], A_t: List[Tensor]) -> torch.nn.Module:
    """
    根据配置创建对应的模型
    """
    if config["doubleskip"]:
        logger.info("使用模型: HybridGNet (双跳连接)")
        model = DoubleSkip(config, D_t, U_t, A_t)
    elif config["skip"]:
        logger.info("使用模型: HybridGNet (单跳连接)")
        model = Skip(config, D_t, U_t, A_t)
    elif not config["pooling"]:
        logger.info("使用模型: HybridGNet (无池化)")
        model = HybridNoPool(config, D_t, U_t, A_t)
    else:
        logger.info("使用模型: HybridGNet (基础版)")
        model = Hybrid(config, D_t, U_t, A_t)
    
    # 加载预训练权重（如果指定）
    if config["load"] != "None" and Path(config["load"]).exists():
        model.load_state_dict(torch.load(config["load"], map_location="cpu"))
        logger.info(f"加载预训练权重: {config['load']}")
    
    return model


def validate_config(config: ConfigType) -> None:
    """
    验证配置参数的有效性
    """
    required_keys = ["name", "inputsize", "epochs", "lr", "stepsize", "gamma", "f", "K"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置缺少必要参数: {key}")
    
    # 验证数值范围
    if config["epochs"] <= 0:
        raise ValueError("训练轮数必须大于0")
    if config["lr"] <= 0:
        raise ValueError("学习率必须大于0")
    if config["batch_size"] <= 0:
        raise ValueError("批次大小必须大于0")


def trainer(train_dataset: LandmarksDataset, val_dataset: LandmarksDataset, 
           model: torch.nn.Module, config: ConfigType) -> None:
    """
    模型训练主函数
    """
    # 固定随机种子
    torch.manual_seed(420)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(420)
    
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 将模型移至设备
    model = model.to(device)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 优化器和学习率调度器
    optimizer = Adam(
        params=model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    
    scheduler = StepLR(
        optimizer,
        step_size=config["stepsize"],
        gamma=config["gamma"]
    )
    
    # 初始化TensorBoard
    log_dir = Path("Training") / config["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 初始化变量
    pool = Pool()
    best_mse = float("inf")
    best_hd = float("inf")
    best_loss = float("inf")
    
    # 训练循环
    logger.info(f"开始训练，共 {config['epochs']} 轮")
    
    for epoch in range(config["epochs"]):
        # 训练阶段
        model.train()
        train_loss_epoch = 0.0
        train_rec_loss_epoch = 0.0
        train_kld_loss_epoch = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for sample_batched in pbar:
            # 数据加载
            image = sample_batched["image"].to(device, non_blocking=True)
            target = sample_batched["landmarks"].to(device, non_blocking=True)
            
            # 前向传播
            out = model(image)
            target_down = pool(target, model.downsample_matrices[0])
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 计算损失
            loss, outloss = calculate_loss(out, target, target_down)
            
            # KL散度损失（如果模型包含）
            kld_loss = torch.tensor(0.0, device=device)
            if hasattr(model, "log_var") and hasattr(model, "mu") and hasattr(model, "kld_weight"):
                kld_loss = -0.5 * torch.mean(1 + model.log_var - model.mu ** 2 - model.log_var.exp())
                loss += model.kld_weight * kld_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 累计损失
            train_loss_epoch += loss.item()
            train_rec_loss_epoch += outloss.item()
            train_kld_loss_epoch += (model.kld_weight * kld_loss).item() if kld_loss.item() > 0 else 0
            
            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "rec_loss": f"{outloss.item():.4f}"
            })
        
        # 计算训练集平均损失
        num_train_batches = len(train_loader)
        train_loss_avg = train_loss_epoch / num_train_batches
        train_rec_loss_avg = train_rec_loss_epoch / num_train_batches
        train_kld_loss_avg = train_kld_loss_epoch / num_train_batches
        
        # 验证阶段
        model.eval()
        val_mse_epoch = 0.0
        val_hd_epoch = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
            for sample_batched in pbar:
                image = sample_batched["image"].to(device, non_blocking=True)
                target = sample_batched["landmarks"].to(device, non_blocking=True)
                
                # 前向传播
                out = model(image)
                if isinstance(out, tuple):
                    out = out[0]  # 只取最终输出
                
                # 重塑输出形状
                out = out.reshape(-1, 2)
                target = target.reshape(-1, 2)
                
                # 计算指标
                hd_dist = hd_landmarks(out, target, config["inputsize"])
                mse = mean_squared_error(out.cpu().numpy(), target.cpu().numpy())
                
                # 累计指标
                val_mse_epoch += mse
                val_hd_epoch += hd_dist
                
                # 更新进度条
                pbar.set_postfix({
                    "mse": f"{mse:.4f}",
                    "hd": f"{hd_dist:.4f}"
                })
        
        # 计算验证集平均指标
        num_val_batches = len(val_loader)
        val_mse_avg = val_mse_epoch / num_val_batches
        val_hd_avg = val_hd_epoch / num_val_batches
        
        # 缩放MSE（与原始代码保持一致）
        mse_scaled = val_mse_avg * 512 * 512
        train_rec_scaled = train_rec_loss_avg * 512 * 512
        
        # 打印日志
        logger.info(f"Epoch [{epoch+1}/{config['epochs']}]")
        logger.info(f"  训练 - 重建损失: {train_rec_scaled:.4f}, KLD损失: {train_kld_loss_avg:.4f}")
        logger.info(f"  验证 - MSE: {mse_scaled:.4f}, HD: {val_hd_avg:.4f}")
        
        # 记录TensorBoard
        writer.add_scalar("Train/Loss", train_loss_avg, epoch)
        writer.add_scalar("Train/KLD_Loss", train_kld_loss_avg, epoch)
        writer.add_scalar("Train/MSE", train_rec_scaled, epoch)
        writer.add_scalar("Validation/MSE", mse_scaled, epoch)
        writer.add_scalar("Validation/Hausdorff_Distance", val_hd_avg, epoch)
        writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)
        
        # 模型保存
        save_path = log_dir
        # 每500轮重置最佳值
        if epoch % 500 == 0:
            best_loss = float("inf")
            best_hd = float("inf")
        
        # 保存最佳MSE模型
        if val_mse_avg < best_mse:
            best_mse = val_mse_avg
            torch.save(model.state_dict(), save_path / "bestMSE.pt")
            logger.info(f"保存最佳MSE模型 (MSE: {mse_scaled:.4f})")
        
        # 保存最佳HD模型
        if val_hd_avg < best_hd:
            best_hd = val_hd_avg
            suffix = f"_{epoch}.pt" if epoch % 500 == 0 else ".pt"
            torch.save(model.state_dict(), save_path / f"bestHD{suffix}")
            logger.info(f"保存最佳HD模型 (HD: {val_hd_avg:.4f})")
        
        # 学习率调度
        scheduler.step()
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 保存最终模型
    torch.save(model.state_dict(), save_path / "final.pt")
    logger.info(f"训练完成，最终模型已保存至: {save_path / 'final.pt'}")
    writer.close()


def main():
    """
    主函数：解析参数、初始化配置、启动训练
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="HybridGNet 训练脚本")
    
    # 基础配置
    parser.add_argument("--name", type=str, required=True, help="实验名称")
    parser.add_argument("--load", default="None", type=str, help="预训练权重路径")
    parser.add_argument("--inputsize", default=1024, type=int, help="输入图像尺寸")
    parser.add_argument("--epochs", default=2500, type=int, help="训练轮数")
    parser.add_argument("--lr", default=1e-4, type=float, help="初始学习率")
    parser.add_argument("--stepsize", default=50, type=int, help="学习率衰减步长")
    parser.add_argument("--gamma", default=0.9, type=float, help="学习率衰减系数")
    
    # 模型参数
    parser.add_argument("--f", default=32, type=int, help="低分辨率滤波器数量")
    parser.add_argument("--K", default=6, type=int, help="K-hops参数")
    parser.add_argument("--layer", default=6, type=int, help="跳连接层数")
    parser.add_argument("--w", default=3, type=int, help="窗口大小")
    parser.add_argument("--l1", default=6, type=int, help="双跳连接第一层")
    parser.add_argument("--l2", default=5, type=int, help="双跳连接第二层")
    
    # 数据配置
    parser.add_argument('--allOrgans', action='store_true', help="是否包含所有器官（锁骨）")
    parser.add_argument('--extended', action='store_true', help="是否使用扩展数据集")
    
    # 模型类型
    parser.add_argument('--skip', action='store_true', help="使用单跳连接模型")
    parser.add_argument('--doubleskip', action='store_true', help="使用双跳连接模型")
    parser.add_argument('--no-pooling', dest='pooling', action='store_false', help="禁用池化")
    
    # 默认参数
    parser.set_defaults(pooling=True, allOrgans=False, extended=False, skip=False, doubleskip=False)
    
    # 解析参数并转换为字典
    args = parser.parse_args()
    config = vars(args)
    
    # 添加额外配置
    config["window"] = (config["w"], config["w"])
    config["latents"] = 64
    config["batch_size"] = 4
    config["val_batch_size"] = 1
    config["weight_decay"] = 1e-5
    config["filters"] = [2, config["f"], config["f"], config["f"], config["f"]//2, config["f"]//2, config["f"]//2]
    
    # 验证配置
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"配置验证失败: {e}")
        return
    
    # 生成矩阵
    logger.info("生成邻接矩阵...")
    if config["allOrgans"]:
        logger.info("器官类型: 肺、心脏、锁骨")
        A, AD, D, U = genMatrixes()
    else:
        logger.info("器官类型: 肺、心脏")
        A, AD, D, U = genMatrixesLH()
    
    # 处理稀疏矩阵
    N1 = A.shape[0]
    N2 = AD.shape[0]
    
    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()
    
    D_ = [D.copy()]
    U_ = [U.copy()]
    
    # 设置节点数和邻接矩阵
    if config["pooling"]:
        logger.info("启用池化")
        config["n_nodes"] = [N1, N1, N1, N2, N2, N2]
        A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
    else:
        logger.info("禁用池化")
        config["n_nodes"] = [N1, N1, N1, N1, N1, N1]
        A_ = [A.copy(), A.copy(), A.copy(), A.copy(), A.copy(), A.copy()]
    
    # 转换为PyTorch稀疏张量
    logger.info("转换矩阵为PyTorch格式...")
    A_t = [scipy_to_torch_sparse(x) for x in A_]
    D_t = [scipy_to_torch_sparse(x) for x in D_]
    U_t = [scipy_to_torch_sparse(x) for x in U_]
    
    # 移动到GPU（如果可用）
    if torch.cuda.is_available():
        A_t = [x.to("cuda:0") for x in A_t]
        D_t = [x.to("cuda:0") for x in D_t]
        U_t = [x.to("cuda:0") for x in U_t]
    
    # 创建数据集
    logger.info("加载数据集...")
    try:
        train_dataset, val_dataset = create_datasets(config)
    except FileNotFoundError as e:
        logger.error(f"数据集加载失败: {e}")
        return
    
    # 创建模型
    logger.info("初始化模型...")
    model = create_model(config, D_t, U_t, A_t)
    
    # 启动训练
    trainer(train_dataset, val_dataset, model, config)


if __name__ == "__main__":
    main()