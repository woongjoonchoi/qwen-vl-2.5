"""
Qwen2.5-VL GUIDE Descriptor Generator
Option B: merged-cell-grid descriptor.

Why Option B (not raw-patch Option A):
  Qwen2.5-VL packs the visual sequence so that each spatial_merge (sms x sms)
  block of raw patches is stored as `spatial_merge_unit = sms**2` CONTIGUOUS
  tokens, and window_index is built on the *merged* grid
  (llm_grid_h = H//sms, llm_grid_w = W//sms).  The authoritative HF expansion is:

      expanded_window_index = window_index_merged[:, None] * smu
                              + arange(smu)[None, :]

  i.e. a raw patch's packed offset is  merged_flat * smu + sub,  NOT  h*W + w.
  Therefore the descriptor grid is the merged grid, and each merged cell expands
  to `smu` contiguous raw tokens.

Descriptor format (INFO_LEN=7 per base record), all spatial fields in MERGED units:
  meta = [
    HEAD_GROUP_NUM,
    head_group_0_offset, ...,
    # at each head-group offset:
    head_start, head_end, case_num,
    # for each case:
    base_case, cur_blocks,
    # for each record (7 ints):
    base_ptr, base_H, base_W, context_count, BLOCK_SIZE, region_H, region_W,
    ...
  ]

For Qwen (single head group, no shift, divisible grid):
  HEAD_GROUP_NUM = 1
  group 0: head_start=0, head_end=num_heads, case_num=1
    case 0: base_case=1
      record 0: base_ptr=0,
                base_H=merged_window, base_W=merged_window,   (merged units, =4)
                context_count=(MH//mw)*(MW//mw),
                BLOCK_SIZE, region_H=MH, region_W=MW           (merged units)

The kernel multiplies by SMU (constexpr) to obtain raw packed offsets; SMU is
NOT stored in meta (it is a model-config constant passed as a constexpr).
"""
import torch

INFO_LEN = 7   # fields per base record


def _cdiv(a, b):
    return (a + b - 1) // b


# ─── Core descriptor builder ──────────────────────────────────────────────────

class QwenDescriptor:
    """
    Builds the flat int32 meta tensor for Qwen2.5-VL window attention,
    operating on the MERGED-cell grid.

    Supports:
      - B=1, T=1
      - merged grid (H//sms, W//sms) divisible by merged_window_size
      - HEAD_GROUP_NUM = 1
    """

    def __init__(self, num_heads: int, raw_window_size: int,
                 spatial_merge_size: int):
        self.num_heads = num_heads
        self.raw_window_size = raw_window_size          # in raw patches (e.g. 8)
        self.spatial_merge_size = spatial_merge_size    # sms (e.g. 2)
        self.smu = spatial_merge_size ** 2              # spatial_merge_unit (e.g. 4)
        self.merged_window = raw_window_size // spatial_merge_size  # e.g. 4

    # -- merged-grid helpers ------------------------------------------------
    def merged_dims(self, H: int, W: int):
        """Return merged-grid (MH, MW) from raw patch dims (H, W)."""
        return H // self.spatial_merge_size, W // self.spatial_merge_size

    def build(self, H: int, W: int, BLOCK_SIZE: int,
              device: str = "cuda") -> torch.Tensor:
        """
        H, W: logical RAW patch grid dimensions (T=1).
        BLOCK_SIZE: raw query tokens processed per Triton program.
        Returns: flat int32 meta tensor.
        """
        MH, MW = self.merged_dims(H, W)
        mw = self.merged_window
        if MH % mw == 0 and MW % mw == 0:
            meta_list = self._build_divisible(MH, MW, BLOCK_SIZE)
        else:
            meta_list = self._build_boundary(MH, MW, BLOCK_SIZE)
        return torch.tensor(meta_list, dtype=torch.int32, device=device)

    def _build_divisible(self, MH: int, MW: int, BLOCK_SIZE: int) -> list:
        """Divisible merged grid: single case, single record."""
        mw = self.merged_window
        smu = self.smu
        n_win = (MH // mw) * (MW // mw)
        tokens_per_window = mw * mw * smu   # raw tokens in one full window (=64)

        # header: HEAD_GROUP_NUM(1) + one group offset ptr
        meta = [1, 2]
        # group 0 descriptor
        meta += [0, self.num_heads, 1]            # head_start, head_end, case_num
        # case 0
        cur_blocks = _cdiv(tokens_per_window, BLOCK_SIZE) * n_win
        meta += [1, cur_blocks]                    # base_case, cur_blocks
        # record 0 (merged units)
        meta += [0, mw, mw, n_win, BLOCK_SIZE, MH, MW]
        return meta

    def _build_boundary(self, MH: int, MW: int, BLOCK_SIZE: int) -> list:
        """
        Non-divisible merged grid: decompose into up to 4 window types.

        case 0: interior full windows   base_H=mw, base_W=mw
        case 1: right boundary strip    base_H=mw, base_W=MW%mw  (if MW%mw != 0)
        case 2: bottom boundary strip   base_H=MH%mw, base_W=mw  (if MH%mw != 0)
        case 3: bottom-right corner     base_H=MH%mw, base_W=MW%mw (if both != 0)

        Each case is a contiguous region (context_count windows).
        Token-to-window mapping is complete and non-overlapping: every merged
        cell (h, w) for h in [0,MH), w in [0,MW) belongs to exactly one case.
        Window order within the descriptor DIFFERS from get_window_index order,
        but the Triton kernel writes each token to its correct output position
        (by packed offset), so correctness is maintained.
        """
        mw = self.merged_window
        smu = self.smu
        H_full = (MH // mw) * mw          # rows covered by full windows
        W_full = (MW // mw) * mw           # cols covered by full windows
        H_rem  = MH - H_full               # boundary height (0 if divisible)
        W_rem  = MW - W_full               # boundary width  (0 if divisible)
        n_h    = MH // mw                  # # of full-height window rows
        n_w    = MW // mw                  # # of full-width  window cols

        # header
        meta = [1, 2]
        meta += [0, self.num_heads]

        cases = []
        # case 0: interior
        if n_h > 0 and n_w > 0:
            n_win = n_h * n_w
            cur_blocks = _cdiv(mw * mw * smu, BLOCK_SIZE) * n_win
            cases.append([1, cur_blocks,
                          0, mw, mw, n_win, BLOCK_SIZE, H_full, W_full])

        # case 1: right boundary strip (full height rows, partial width)
        if W_rem > 0 and n_h > 0:
            n_win = n_h
            cur_blocks = _cdiv(mw * W_rem * smu, BLOCK_SIZE) * n_win
            # base_ptr: first row, column W_full; region_W = W_rem (single column of windows)
            cases.append([1, cur_blocks,
                          W_full, mw, W_rem, n_win, BLOCK_SIZE, H_full, W_rem])

        # case 2: bottom boundary strip (partial height rows, full width)
        if H_rem > 0 and n_w > 0:
            n_win = n_w
            cur_blocks = _cdiv(H_rem * mw * smu, BLOCK_SIZE) * n_win
            # base_ptr: row H_full, column 0; region_H = H_rem, region_W = W_full
            cases.append([1, cur_blocks,
                          H_full * MW, H_rem, mw, n_win, BLOCK_SIZE, H_rem, W_full])

        # case 3: bottom-right corner
        if H_rem > 0 and W_rem > 0:
            cur_blocks = _cdiv(H_rem * W_rem * smu, BLOCK_SIZE)
            cases.append([1, cur_blocks,
                          H_full * MW + W_full, H_rem, W_rem, 1, BLOCK_SIZE, H_rem, W_rem])

        meta.append(len(cases))
        for c in cases:
            meta += c
        return meta

    def per_batch_blocks(self, H: int, W: int, BLOCK_SIZE: int) -> int:
        meta_list = self.build(H, W, BLOCK_SIZE, device="cpu").tolist()
        return _compute_per_batch_blocks(meta_list, self.smu)

    def emit_offsets(self, H: int, W: int, BLOCK_SIZE: int,
                     device: str = "cpu") -> torch.Tensor:
        """
        Emit raw packed offsets in the order the kernel visits them.
        Equivalent to Qwen's expanded_window_index for divisible grids.
        Returns: [S_raw] int64.
        """
        MH, MW = self.merged_dims(H, W)
        meta_list = self.build(H, W, BLOCK_SIZE, device="cpu").tolist()
        return _emit_offsets_from_meta(meta_list, MH, MW, self.smu)

    def emit_window_groups(self, H: int, W: int, BLOCK_SIZE: int) -> list:
        """Return list of windows, each a list of raw packed offsets."""
        offsets = self.emit_offsets(H, W, BLOCK_SIZE).tolist()
        tpw = self.merged_window * self.merged_window * self.smu  # 64
        return [offsets[i:i + tpw] for i in range(0, len(offsets), tpw)]


# ─── Helper functions ─────────────────────────────────────────────────────────

def _compute_per_batch_blocks(meta: list, smu: int) -> int:
    """Compute per_batch_blocks from a meta list (token counts include smu)."""
    HEAD_GROUP_NUM = meta[0]
    per_batch_blocks = 0
    for g in range(HEAD_GROUP_NUM):
        grp_off = meta[1 + g]
        case_num = meta[grp_off + 2]
        base_offset = grp_off + 3
        for _ in range(case_num):
            base_case = meta[base_offset]
            case_blocks = 0
            for j in range(base_case):
                bH  = meta[base_offset + 1 + INFO_LEN * j + 2]
                bW  = meta[base_offset + 1 + INFO_LEN * j + 3]
                ctx = meta[base_offset + 1 + INFO_LEN * j + 4]
                BS  = meta[base_offset + 1 + INFO_LEN * j + 5]
                case_blocks = max(case_blocks, _cdiv(bH * bW * smu, BS) * ctx)
            meta[base_offset + 1] = case_blocks
            per_batch_blocks += case_blocks
            base_offset += INFO_LEN * base_case + 2
    return per_batch_blocks


def _emit_offsets_from_meta(meta: list, MH: int, MW: int, smu: int) -> torch.Tensor:
    """
    Emit raw packed offsets in kernel traversal order.
    base_ptr is an absolute flat index in the full merged grid (h*MW + w),
    just like the original Swin kernel. region_W is used only to determine
    the context-grid stride (n_ctx_W), NOT to decode (h_base, w_base).
    """
    HEAD_GROUP_NUM = meta[0]
    offsets = []
    for g in range(HEAD_GROUP_NUM):
        grp_off = meta[1 + g]
        case_num = meta[grp_off + 2]
        base_offset = grp_off + 3
        for _ in range(case_num):
            base_case = meta[base_offset]
            for j in range(base_case):
                base_ptr = meta[base_offset + 1 + INFO_LEN * j + 1]
                bH       = meta[base_offset + 1 + INFO_LEN * j + 2]
                bW       = meta[base_offset + 1 + INFO_LEN * j + 3]
                ctx      = meta[base_offset + 1 + INFO_LEN * j + 4]
                rgn_W    = meta[base_offset + 1 + INFO_LEN * j + 7]
                n_ctx_W  = rgn_W // bW if bW > 0 else 1
                # base_ptr is absolute in the full merged grid (width=MW)
                h_base   = base_ptr // MW
                w_base   = base_ptr %  MW
                for c in range(ctx):
                    ctx_h = c // n_ctx_W
                    ctx_w = c %  n_ctx_W
                    h_off = ctx_h * bH
                    w_off = ctx_w * bW
                    for m_row in range(bH):
                        for m_col in range(bW):
                            merged_flat = (h_base + h_off + m_row) * MW \
                                          + (w_base + w_off + m_col)
                            for sub in range(smu):
                                offsets.append(merged_flat * smu + sub)
            base_offset += INFO_LEN * base_case + 2
    return torch.tensor(offsets, dtype=torch.int64)


# ─── Convenience wrappers ──────────────────────────────────────────────────────

def build_qwen_descriptor(H, W, num_heads, raw_window_size, spatial_merge_size,
                          BLOCK_SIZE, device="cuda"):
    gen = QwenDescriptor(num_heads, raw_window_size, spatial_merge_size)
    return gen.build(H, W, BLOCK_SIZE, device=device)


def get_per_batch_blocks(H, W, num_heads, raw_window_size, spatial_merge_size,
                         BLOCK_SIZE):
    gen = QwenDescriptor(num_heads, raw_window_size, spatial_merge_size)
    return gen.per_batch_blocks(H, W, BLOCK_SIZE)
