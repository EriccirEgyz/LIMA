"""补测: 8B @ 无 grad ckpt 的显存随序列长度增长斜率, 外推 32K 是否放得下单卡。
GPU7 与训练共卡(~91G 空闲): 长度递增试跑, OOM 即停, 用成功的点拟合线性外推。"""
import torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
from transformers import AutoModelForCausalLM

MODEL = "/mnt/beegfs/workspace/scy/yizhigao/LIMA/models/Qwen3-8B"
apply_liger_kernel_to_qwen3(fused_linear_cross_entropy=True, cross_entropy=False)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.cuda()
model.train()  # 关键: 不开 gradient checkpointing

peaks = []
for S in (8192, 12288, 16384, 20480, 24576, 32768):
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(0, 151936, (1, S), device="cuda")
        out = model(input_ids=ids, labels=ids.clone())
        out.loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        peaks.append((S, peak))
        print(f"S={S:6d}  OK    peak={peak:6.1f}G  (其中激活≈{peak - 32:.1f}G, 32G=模型16G+梯度16G)")
        model.zero_grad(set_to_none=True)
        del out, ids
    except torch.cuda.OutOfMemoryError:
        print(f"S={S:6d}  OOM   (该卡当前仅 ~91G 空闲, 训练占 ~50G)")
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

if len(peaks) >= 2:
    (s1, p1), (s2, p2) = peaks[-2], peaks[-1]  # 用最靠右两点拟合(含全部常量后更准)
    slope = (p2 - p1) / ((s2 - s1) / 1024)
    pred32k = p2 + slope * (32768 - s2) / 1024
    print(f"\n实测斜率 ≈ {slope:.2f} G / 1K tokens")
    print(f"外推 32K 无 ckpt 峰值 ≈ {pred32k:.0f}G  vs  单卡上限 141G  ->  {'放得下' if pred32k < 141 else '放不下'}")
