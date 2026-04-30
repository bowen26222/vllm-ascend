from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.forward_context import get_forward_context, set_forward_context


@dataclass
class _ToyAttentionMetadata:
    # Minimal subset mimicking vLLM attention metadata with query_start_loc
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


def _make_mock_vllm_config() -> MagicMock:
    vllm_config = MagicMock()
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.parallel_config.data_parallel_rank = 0
    return vllm_config


def _make_attn(num_reqs: int, lens: list[int]) -> _ToyAttentionMetadata:
    assert len(lens) == num_reqs
    qsl = [0]
    for l in lens:
        qsl.append(qsl[-1] + l)
    total = qsl[-1]

    return _ToyAttentionMetadata(
        num_actual_tokens=total,
        max_query_len=max(lens) if lens else 0,
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
        max_seq_len=max(lens) if lens else 0,
        seq_lens=torch.tensor(lens, dtype=torch.int32),
        block_table=torch.arange(num_reqs * 4, dtype=torch.int32).reshape(
            num_reqs, 4
        ),
        slot_mapping=torch.arange(total, dtype=torch.int32),
    )


def test_split_wrapper_skips_when_disabled():
    vllm_config = _make_mock_vllm_config()

    # Split disabled
    split_cfg = MagicMock()
    split_cfg.enabled = False
    split_cfg.enable_parallel_streams = False
    split_cfg.num_splits = 2
    split_cfg.min_batch_size_for_split = 2

    attn = _make_attn(num_reqs=4, lens=[1, 2, 1, 2])

    called = {"n": 0}

    def runnable(*, input_ids, positions, **kwargs):
        called["n"] += 1
        # should see original forward context
        ctx = get_forward_context()
        meta = next(iter(ctx.attn_metadata.values()))
        assert meta.query_start_loc.numel() == 5
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
            device=torch.device("cpu"),
        )

        input_ids = torch.arange(attn.num_actual_tokens, dtype=torch.int64)
        positions = torch.arange(attn.num_actual_tokens, dtype=torch.int64)

        with set_forward_context(attn_metadata={"l0": attn}, vllm_config=vllm_config):
            out = wrapper.forward(input_ids=input_ids, positions=positions)

        assert torch.equal(out, input_ids)
        assert called["n"] == 1


def test_split_wrapper_splits_and_preserves_output():
    vllm_config = _make_mock_vllm_config()

    split_cfg = MagicMock()
    split_cfg.enabled = True
    split_cfg.enable_parallel_streams = False
    split_cfg.num_splits = 2
    split_cfg.min_batch_size_for_split = 2

    attn = _make_attn(num_reqs=4, lens=[2, 1, 3, 2])

    calls: list[tuple[int, int]] = []

    def runnable(*, input_ids, positions, **kwargs):
        ctx = get_forward_context()
        meta = next(iter(ctx.attn_metadata.values()))

        # query_start_loc should be shifted to start at 0 per split
        assert int(meta.query_start_loc[0].item()) == 0
        assert meta.num_actual_tokens == input_ids.shape[0]
        calls.append((int(meta.query_start_loc.numel() - 1), input_ids.shape[0]))
        return input_ids.clone()

    with patch("vllm_ascend.worker.npu_split_wrapper.get_ascend_config") as get_cfg:
        asc = MagicMock()
        asc.split_batch_config = split_cfg
        get_cfg.return_value = asc

        from vllm_ascend.worker.npu_split_wrapper import AscendSplitBatchWrapper

        wrapper = AscendSplitBatchWrapper(
            runnable=runnable,
            vllm_config=vllm_config,
            runtime_mode=MagicMock(),
            device=torch.device("cpu"),
        )

        input_ids = torch.arange(attn.num_actual_tokens, dtype=torch.int64)
        positions = torch.arange(attn.num_actual_tokens, dtype=torch.int64)

        with set_forward_context(attn_metadata={"l0": attn}, vllm_config=vllm_config):
            out = wrapper.forward(input_ids=input_ids, positions=positions)

        assert torch.equal(out, input_ids)
        # Should split into 2 calls (2 reqs each)
        assert len(calls) == 2
        assert calls[0][0] == 2 and calls[1][0] == 2
        assert sum(t for _, t in calls) == attn.num_actual_tokens


def test_split_wrapper_skips_when_ubatching_present():
    vllm_config = _make_mock_vllm_config()

    split_cfg = MagicMock()
    split_cfg.enabled = True
    split_cfg.enable_parallel_streams = False
    split_cfg.num_splits = 2
    split_cfg.min_batch_size_for_split = 2

    attn = _make_attn(num_reqs=4, lens=[1, 1, 1, 1])

    called = {"n": 0}

    def runnable(*, input_ids, positions, **kwargs):
        called["n"] += 1
        return input_ids

    with patch("vllm_ascend.worker.npu_split_wrapper.get_ascend_config") as get_cfg:
        asc = MagicMock()
        asc.split_batch_config = split_cfg
        get_cfg.return_value = asc

        from vllm.v1.worker.ubatch_utils import UBatchSlice
        from vllm_ascend.worker.npu_split_wrapper import AscendSplitBatchWrapper

        wrapper = AscendSplitBatchWrapper(
            runnable=runnable,
            vllm_config=vllm_config,
            runtime_mode=MagicMock(),
            device=torch.device("cpu"),
        )

        input_ids = torch.arange(attn.num_actual_tokens, dtype=torch.int64)
        positions = torch.arange(attn.num_actual_tokens, dtype=torch.int64)

        ubatch_slices = [
            UBatchSlice(slice(0, 2), slice(0, 2)),
            UBatchSlice(slice(2, 4), slice(2, 4)),
        ]

        with set_forward_context(
            attn_metadata={"l0": attn},
            vllm_config=vllm_config,
            ubatch_slices=ubatch_slices,
        ):
            out = wrapper.forward(input_ids=input_ids, positions=positions)

        assert torch.equal(out, input_ids)
        assert called["n"] == 1


def test_split_wrapper_interface_has_split_and_execute():
    vllm_config = _make_mock_vllm_config()

    split_cfg = MagicMock()
    split_cfg.enabled = True
    split_cfg.enable_parallel_streams = False
    split_cfg.num_splits = 2
    split_cfg.min_batch_size_for_split = 2

    with patch("vllm_ascend.worker.npu_split_wrapper.get_ascend_config") as get_cfg:
        asc = MagicMock()
        asc.split_batch_config = split_cfg
        get_cfg.return_value = asc

        from vllm_ascend.worker.npu_split_wrapper import AscendSplitBatchWrapper

        wrapper = AscendSplitBatchWrapper(
            runnable=lambda **kw: kw["input_ids"],
            vllm_config=vllm_config,
            runtime_mode=MagicMock(),
            device=torch.device("cpu"),
        )

        assert hasattr(wrapper, "split_and_execute")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
