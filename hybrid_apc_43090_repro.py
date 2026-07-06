"""
hybrid_apc_43090_repro.py
=========================================================================
逻辑等价复现 —— vLLM issue #43090
(granite hybrid 模型开 prefix caching 后 cached vs uncached 输出发散)

目的:在纯 Windows / 纯 CPU / 无 vLLM 环境下,证明 hybrid 模型
     "attention 组命中 + mamba 组 miss" 时,调度层会让 mamba 拿到一个
     错误的 has_initial_states=True(实际并没有 cached state),
     这正是端到端输出发散的根因。

范围声明(诚实边界):
  - 本脚本复现的是【调度逻辑链】,逻辑摘取自:
      * vllm/v1/core/kv_cache_coordinator.py
          HybridKVCacheCoordinator.find_longest_cache_hit  (含 #45238 的 continue)
      * vllm/v1/core/single_type_kv_cache_manager.py
          MambaManager.find_longest_cache_hit  (只认最后一个对齐的 cached block)
      * vllm/v1/core/kv_cache_manager.py
          num_local_computed_tokens = num_computed + num_new_computed(标量,不分组)
      * vllm/v1/attention/backends/mamba_attn.py  (line ~500)
          has_initial_states_p = (num_computed_tokens > 0)
  - 本脚本【不是】vLLM 端到端跑出的数值发散(那一环需 Linux GPU),
    第 4 环(mamba kernel 用空 state 续算 -> logits 发散)需真实环境复核。
=========================================================================
"""

from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# 极简 block pool:只记录哪些 (组, block_hash) 被缓存过
# --------------------------------------------------------------------------
class FakeBlockPool:
    def __init__(self):
        self.cached: set[tuple[str, int]] = set()

    def cache(self, group: str, block_hash: int) -> None:
        self.cached.add((group, block_hash))

    def is_cached(self, group: str, block_hash: int) -> bool:
        return (group, block_hash) in self.cached


# --------------------------------------------------------------------------
# Full-Attention 组命中:从左到右,连续命中已缓存的 block(标准前缀命中)
# 返回命中的 block 数量
# --------------------------------------------------------------------------
def full_attn_find_hit(block_hashes, max_blocks, pool: FakeBlockPool) -> int:
    n = 0
    for i in range(max_blocks):
        if pool.is_cached("attn", block_hashes[i]):
            n += 1
        else:
            break
    return n


# --------------------------------------------------------------------------
# Mamba 组命中(摘自 MambaManager.find_longest_cache_hit):
#   从右往左找【最后一个】被缓存的 block,且必须落在对齐边界:
#       (i + 1) * mamba_block_size % alignment_tokens == 0
#   Mamba 只有单点 state checkpoint,所以要么命中到某个对齐点,要么整组 miss。
# 返回命中的 block 数量(0 表示 miss)
# --------------------------------------------------------------------------
def mamba_find_hit(block_hashes, max_blocks, mamba_block_size,
                   alignment_tokens, pool: FakeBlockPool) -> int:
    for i in range(max_blocks - 1, -1, -1):
        if pool.is_cached("mamba", block_hashes[i]):
            if (mamba_block_size != alignment_tokens
                    and (i + 1) * mamba_block_size % alignment_tokens != 0):
                continue  # 不在对齐边界,跳过
            return i + 1  # 命中到第 i 个 block(单点 checkpoint)
    return 0


# --------------------------------------------------------------------------
# HybridKVCacheCoordinator.find_longest_cache_hit 的收敛逻辑(simple hybrid:
# 1 个 full-attn 组 + 1 个 mamba 组)。
#   apply_45238_fix=True  -> 复现你 PR #47491 的行为(mamba miss 时 continue,
#                            保留 attention hit)
#   apply_45238_fix=False -> 复现 #45238 之前的旧行为(mamba miss 清零所有 hit)
# 返回 (hit_length_tokens, attn_hit_blocks, mamba_hit_blocks)
# --------------------------------------------------------------------------
def coordinator_find_longest_cache_hit(
    block_hashes, max_len_tokens, attn_block_size, mamba_block_size,
    alignment_tokens, pool: FakeBlockPool, apply_45238_fix: bool,
):
    # --- full attention 先扫(vLLM 里 full-attn 组总是排第一)---
    attn_max_blocks = max_len_tokens // attn_block_size
    attn_hit_blocks = full_attn_find_hit(block_hashes, attn_max_blocks, pool)
    attn_hit_len = attn_hit_blocks * attn_block_size

    curr_hit_length = attn_hit_len

    # --- mamba 组 ---
    mamba_max_blocks = max_len_tokens // mamba_block_size
    mamba_hit_blocks = mamba_find_hit(
        block_hashes, mamba_max_blocks, mamba_block_size, alignment_tokens, pool
    )
    mamba_hit_len = mamba_hit_blocks * mamba_block_size

    if mamba_hit_len == 0:
        if apply_45238_fix:
            # ===== #45238 (PR #47491) 的修复:=====
            # "Don't let a Mamba miss zero out valid attention cache hits."
            # continue -> 保留 attention 的 curr_hit_length,mamba blocks 留空
            mamba_result_blocks = 0
            # curr_hit_length 保持 = attn_hit_len  (> 0)
        else:
            # ===== #45238 之前的旧行为:mamba miss 把 hit 清零 =====
            curr_hit_length = 0
            mamba_result_blocks = 0
    else:
        # mamba 命中:取两组交集(向下取整到 mamba 边界)
        curr_hit_length = min(curr_hit_length, mamba_hit_len)
        mamba_result_blocks = mamba_hit_blocks

    return curr_hit_length, attn_hit_blocks, mamba_result_blocks


# --------------------------------------------------------------------------
# 上层 + mamba metadata 判定
# --------------------------------------------------------------------------
@dataclass
class Outcome:
    label: str
    hit_length: int
    attn_hit_blocks: int
    mamba_hit_blocks: int
    num_computed_tokens: int
    has_initial_states: bool
    mamba_actually_has_state: bool
    bug: bool = field(init=False)

    def __post_init__(self):
        # BUG 判据:mamba 被告知"有 initial state",但它其实没有任何 cached state
        self.bug = self.has_initial_states and not self.mamba_actually_has_state


def simulate(apply_45238_fix: bool, pool, block_hashes, max_len_tokens,
             attn_block_size, mamba_block_size, alignment_tokens, label):
    hit_length, attn_blocks, mamba_blocks = coordinator_find_longest_cache_hit(
        block_hashes, max_len_tokens, attn_block_size, mamba_block_size,
        alignment_tokens, pool, apply_45238_fix,
    )

    # kv_cache_manager.py: 标量,应用到所有组
    num_computed_tokens = hit_length

    # mamba_attn.py (~line 500): has_initial_states_p = num_computed_tokens > 0
    has_initial_states = num_computed_tokens > 0

    # mamba 组这次到底有没有拿到 cached state?
    mamba_actually_has_state = mamba_blocks > 0

    return Outcome(
        label=label,
        hit_length=hit_length,
        attn_hit_blocks=attn_blocks,
        mamba_hit_blocks=mamba_blocks,
        num_computed_tokens=num_computed_tokens,
        has_initial_states=has_initial_states,
        mamba_actually_has_state=mamba_actually_has_state,
    )


def print_outcome(o: Outcome):
    print(f"  [{o.label}]")
    print(f"    coordinator returned hit_length   = {o.hit_length} tokens")
    print(f"    attention cache-hit blocks        = {o.attn_hit_blocks}")
    print(f"    mamba cache-hit blocks            = {o.mamba_hit_blocks}")
    print(f"    scheduler num_computed_tokens     = {o.num_computed_tokens}")
    print(f"    mamba.has_initial_states          = {o.has_initial_states}")
    print(f"    mamba actually has cached state?  = {o.mamba_actually_has_state}")
    verdict = ">>> BUG: mamba told it HAS initial state, but it does NOT " \
              "-> recurs from empty state -> output diverges" \
        if o.bug else "    OK : has_initial_states matches real state"
    print(f"    {verdict}")
    print()


def main():
    # ---- 构造场景 ----
    # 8 个逻辑块;attention block_size = mamba block_size = 16;
    # scheduler(alignment) block_size = 32(LCM,常见于 hybrid)。
    # attention 命中前 6 个 block;
    # mamba 的 state checkpoint 落在 request-unique token(第 7、8 块),
    #   前 6 块的 mamba 侧未被缓存 -> mamba 整组 miss。
    ATTN_BS = 16
    MAMBA_BS = 16
    ALIGNMENT = 32
    NUM_BLOCKS = 8
    MAX_LEN = NUM_BLOCKS * ATTN_BS  # 128 tokens

    block_hashes = list(range(1000, 1000 + NUM_BLOCKS))  # 唯一 hash

    pool = FakeBlockPool()
    # attention 侧:前 6 个 block 命中
    for i in range(6):
        pool.cache("attn", block_hashes[i])
    # mamba 侧:前 6 块都没有 cached state(checkpoint 落在别处)-> miss
    #   (故意一个都不 cache,模拟 "checkpoint lands in request-unique tokens")

    print("=" * 70)
    print("vLLM issue #43090 -- hybrid prefix caching root-cause repro (logic-level)")
    print("Scenario: attention group hits first 6 blocks (96 tokens), mamba group misses")
    print("=" * 70)
    print()

    old = simulate(False, pool, block_hashes, MAX_LEN, ATTN_BS, MAMBA_BS,
                   ALIGNMENT, "pre-#45238 (mamba miss zeros the hit)")
    new = simulate(True, pool, block_hashes, MAX_LEN, ATTN_BS, MAMBA_BS,
                   ALIGNMENT, "#45238 / PR #47491 (preserve attention hit)")

    print_outcome(old)
    print_outcome(new)

    print("=" * 70)
    print("Conclusion:")
    print(f"  pre-#45238  bug={old.bug}  ->  mamba recomputes from scratch, CORRECT "
          "(hit rate 0, just slower)")
    print(f"  #45238 fix  bug={new.bug}  ->  mamba wrongly gets has_initial_states=True")
    print("                          with NO cached state -> empty-state recurrence -> diverge")
    print()
    print("  Root cause: has_initial_states = (num_computed_tokens > 0) only looks at")
    print("  the global scalar; it does NOT distinguish WHICH group produced the hit.")
    print("  The attention hit inflates num_computed_tokens, so mamba is misjudged")
    print("  as having an initial state.")
    print("=" * 70)


if __name__ == "__main__":
    main()
