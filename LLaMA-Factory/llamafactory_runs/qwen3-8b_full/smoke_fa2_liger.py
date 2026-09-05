"""冒烟: Qwen3-8B + FlashAttention-2 + Liger fused CE + grad ckpt @ 32K, 单卡(空闲卡)。
验证三件套在本 env 可用; 顺带测 32K fwd+bwd 耗时(对照在跑 job 的 ~17.4s/microstep@19.5K均值)。"""
import time

import torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/beegfs/workspace/scy/yizhigao/LIMA/models/Qwen3-8B"

apply_liger_kernel_to_qwen3(fused_linear_cross_entropy=True, cross_entropy=False)
print("liger applied to qwen3 (fused linear CE)")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
print("attn_implementation =", model.config._attn_implementation)
model.cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()

S = 32768
ids = torch.randint(0, 151936, (1, S), device="cuda")
labels = ids.clone()

out = model(input_ids=ids, labels=labels)  # warmup
out.loss.backward()
model.zero_grad(set_to_none=True)

torch.cuda.synchronize()
t0 = time.time()
N = 3
for _ in range(N):
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
dt = (time.time() - t0) / N
print(
    f"loss={out.loss.item():.4f}  fwd+bwd(32K, grad-ckpt) = {dt:.2f}s/microstep"
    f"  peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G"
)
