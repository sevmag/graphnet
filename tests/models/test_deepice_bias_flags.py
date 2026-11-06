"""Tests that relative-attention blocks always carry their bias."""

import pytest

from graphnet.models.components.attention_blocks import Attention_rel
from graphnet.models.gnn import DeepIce


def test_relative_attention_without_a_bias_is_rejected() -> None:
    """Attention without the bias belongs in `Block`, which is far cheaper."""
    with pytest.raises(ValueError, match="use Block"):
        Attention_rel(64, use_attn_bias=False, use_activation_bias=False)
    with pytest.raises(ValueError, match="use Block"):
        DeepIce(
            hidden_dim=64,
            seq_length=32,
            depth=0,
            head_size=32,
            depth_rel=2,
            include_dynedge=False,
            n_features=6,
            use_attn_bias=False,
            use_activation_bias=False,
        )


def test_alibi_needs_the_logit_bias() -> None:
    """ALiBi has no per-pair vector, so the logits are its only use."""
    with pytest.raises(ValueError, match="use Block"):
        Attention_rel(64, use_attn_bias=False, alibi=True)


@pytest.mark.parametrize("attn, activation", [(True, False), (False, True)])
def test_either_use_of_the_bias_is_accepted(
    attn: bool, activation: bool
) -> None:
    """Each use on its own is still a valid ablation."""
    Attention_rel(64, use_attn_bias=attn, use_activation_bias=activation)
