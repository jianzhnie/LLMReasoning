#!/usr/bin/env python3
"""
distributed_comm_test.py

功能：分布式通信测试脚本（支持 NPU/GPU）
用途：验证 HCCL/NCCL 通信是否正常
运行：torchrun --nproc-per-node=8 run_single_node.sh ...
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

# 尝试导入 NPU 支持（Ascend 环境）
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def get_device(local_rank: int) -> torch.device:
    """
    根据可用设备返回正确的 torch.device。
    优先使用 NPU（Ascend），否则使用 CUDA。
    """
    if HAS_NPU:
        assert hasattr(torch, 'npu'), 'torch_npu 扩展未正确加载'
        return torch.device(f'npu:{local_rank}')
    else:
        return torch.device(f'cuda:{local_rank}')


def set_device(local_rank: int):
    """为当前进程设置默认设备"""
    if HAS_NPU:
        torch.npu.set_device(local_rank)  # ✅ 正确方式
    else:
        torch.cuda.set_device(local_rank)


def get_backend() -> str:
    """根据硬件自动选择合适的分布式后端"""
    if HAS_NPU:
        return 'nccl'
    else:
        return 'nccl'


# ========================================
# 日志与辅助函数
# ========================================


def log_rank(msg: str, rank=None):
    """带 Rank 前缀的日志输出"""
    rank_str = f'{rank:2d}' if rank is not None else os.environ.get(
        'RANK', 'unknown')
    print(f'[Rank {rank_str}] {msg}', flush=True)


def validate_tensor(tensor: torch.Tensor, expected: float, op_name: str):
    """验证通信结果"""
    if abs(tensor.item() - expected) < 1e-5:
        log_rank(f'✅ {op_name} 成功：结果正确。', tensor.device.index)
    else:
        log_rank(f'❌ {op_name} 失败：期望 {expected:.1f}，实际 {tensor.item():.1f}',
                 tensor.device.index)


# ========================================
# 分布式核心逻辑
# ========================================


def run(rank: int, world_size: int, local_rank: int):
    """分布式训练核心逻辑"""
    device = get_device(local_rank)

    # 测试 1: AllReduce (Sum)
    tensor = torch.tensor([float(rank)], dtype=torch.float32, device=device)
    log_rank(
        f'Local Rank {local_rank} | Device: {device} | 初始值: {tensor.item():.1f}',
        rank)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    log_rank(f'AllReduce 结果: {tensor.item():.1f}', rank)
    expected = world_size * (world_size - 1) / 2
    validate_tensor(tensor, expected, 'AllReduce')
    dist.barrier()

    # 测试 2: Broadcast (rank 0 发送)
    if rank == 0:
        broadcast_tensor = torch.tensor([123.0], device=device)
    else:
        broadcast_tensor = torch.zeros(1, device=device)

    dist.broadcast(broadcast_tensor, src=0)
    log_rank(f'Broadcast 后值: {broadcast_tensor.item():.1f}', rank)
    validate_tensor(broadcast_tensor, 123.0, 'Broadcast')
    dist.barrier()

    # 测试 3: AllGather
    gather_list = [
        torch.zeros(1, dtype=torch.float32, device=device)
        for _ in range(world_size)
    ]
    dist.all_gather(gather_list, tensor)
    gathered_values = [
        gathered_tensor.item() for gathered_tensor in gather_list
    ]
    log_rank(f'AllGather 结果: {gathered_values}', rank)
    validate_tensor(gather_list[0], sum(range(world_size)),
                    'AllGather')  # 验证第一个元素
    dist.barrier()

    # 测试 4: ReduceScatter
    scatter_list = [
        torch.tensor([float(rank)], dtype=torch.float32, device=device)
        for _ in range(world_size)
    ]
    output = torch.zeros(1, dtype=torch.float32, device=device)
    dist.reduce_scatter(output, scatter_list, op=dist.ReduceOp.SUM)
    log_rank(f'ReduceScatter 结果: {output.item():.1f}', rank)
    validate_tensor(output, sum(range(world_size)), 'ReduceScatter')

    log_rank('🎉 所有通信测试完成！', rank)


def setup_distributed() -> tuple[int, int, int]:
    env = os.environ
    try:
        rank = int(env['RANK'])
        local_rank = int(env.get('LOCAL_RANK', rank))
        world_size = int(env['WORLD_SIZE'])
    except KeyError as e:
        raise EnvironmentError(f"缺少环境变量: {e}. 请使用 'torchrun' 启动。") from e
    except ValueError as e:
        raise ValueError(f'环境变量格式错误: {e}') from e

    backend = get_backend()
    master_addr = env.get('MASTER_ADDR', 'unknown')
    master_port = env.get('MASTER_PORT', 'unknown')

    log_rank(
        f"正在初始化进程组 backend='{backend.upper()}' "
        f'MASTER={master_addr}:{master_port}', rank)

    dist.init_process_group(backend=backend,
                            rank=rank,
                            world_size=world_size,
                            timeout=timedelta(seconds=60))

    # ✅ 正确设置设备
    set_device(local_rank)

    if dist.is_initialized():
        log_rank(
            f'分布式环境初始化完成。World Size={world_size}, Local Rank={local_rank}',
            rank)
        dist.barrier()

    return rank, world_size, local_rank


def cleanup():
    """安全清理分布式环境"""
    if dist.is_initialized():
        dist.barrier()  # 确保所有 rank 都到达此处
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        rank, world_size, local_rank = setup_distributed()
        run(rank, world_size, local_rank)
    except KeyboardInterrupt:
        print(f"[Rank {os.environ.get('RANK', 'unknown')}] 被用户中断。",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        rank_str = os.environ.get('RANK', 'unknown')
        print(f'[Rank {rank_str}] 发生未预期错误: {type(e).__name__}: {e}',
              file=sys.stderr)
        raise  # 保留原始 traceback
    finally:
        cleanup()
