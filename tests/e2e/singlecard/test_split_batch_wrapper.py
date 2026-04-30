import pytest
import torch

from vllm.forward_context import get_forward_context, set_forward_context


@pytest.mark.skipif(not hasattr(torch, "npu"), reason="requires torch.npu")
def test_split_batch_wrapper_e2e_smoke():
    """A lightweight e2e smoke test under the e2e harness.

    This does not load a real model; it validates that the split wrapper can
    run inside the e2e environment where torch.npu is available.
    """
    from dataclasses import dataclass
    from unittest.mock import MagicMock, patch

    @dataclass
    class ToyMeta:
        num_actual_tokens: int
        max_query_len: int
        query_start_loc: torch.Tensor
        max_seq_len: int
        seq_lens: torch.Tensor
        block_table: torch.Tensor
        slot_mapping: torch.Tensor

    qsl = torch.tensor([0, 2, 3, 6, 8], dtype=torch.int32)
    seq_lens = torch.tensor([2, 1, 3, 2], dtype=torch.int32)
    meta = ToyMeta(
        num_actual_tokens=8,
        max_query_len=3,
        query_start_loc=qsl,
        max_seq_len=3,
        seq_lens=seq_lens,
        block_table=torch.zeros((4, 4), dtype=torch.int32),
        slot_mapping=torch.arange(8, dtype=torch.int32),
    )

    vllm_config = MagicMock()
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.parallel_config.data_parallel_rank = 0

    split_cfg = MagicMock()
    split_cfg.enabled = True
    split_cfg.enable_parallel_streams = False
    split_cfg.num_splits = 2
    split_cfg.min_batch_size_for_split = 2

    seen = {"calls": 0}

    def runnable(*, input_ids, positions, **kwargs):
        ctx = get_forward_context()
        m = next(iter(ctx.attn_metadata.values()))
        assert int(m.query_start_loc[0].item()) == 0
        seen["calls"] += 1
        return input_ids

    with patch("vllm_ascend.worker.npu_split_wrapper.get_ascend_config") as get_cfg:
        asc = MagicMock()
        asc.split_batch_config = split_cfg
        get_cfg.return_value = asc

        from vllm_ascend.worker.npu_split_wrapper import AscendSplitBatchWrapper

        wrapper = AscendSplitBatchWrapper(
            runnable=runnable,
            vllm_config=vllm_config,
            runtime_mode=MagicMock(),
            device=torch.device("npu"),
        )

        input_ids = torch.arange(8, dtype=torch.int64, device="cpu")
        positions = torch.arange(8, dtype=torch.int64, device="cpu")

        with set_forward_context(attn_metadata={"l0": meta}, vllm_config=vllm_config):
            out = wrapper.forward(input_ids=input_ids, positions=positions)

        assert out.shape[0] == 8
        assert seen["calls"] == 2
