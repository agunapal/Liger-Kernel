from typing import Callable, Dict, List, Optional, Union

import torch
from torchtune.modules.attention_utils import _MaskType


def decoder_forward(
    self,
    tokens: torch.Tensor,
    *,
    mask: Optional[_MaskType] = None,
    encoder_input: Optional[torch.Tensor] = None,
    encoder_mask: Optional[torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Args:
        tokens (torch.Tensor): input tensor with shape ``[b x s]``
        mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
            and before the softmax. This parameter is required during inference if caches have been setup.
            Either:

            A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
            or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
            A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
            token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
            is used by default.

            A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
            created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
            :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
            Default is None.
        encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
        encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
            tokens and encoder embeddings. A True value at position ``i,j`` means token ``i`` can attend
            to embedding ``j`` in the decoder. Mask has shape ``[b x s x s_e]``. Default is None,
            but this is required during inference if the model has been setup with any layers
            which use encoder embeddings and caches have been setup.
        input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
            of each token. During training, this is used to indicate the positions
            of each token relative to its sample when packed, shape ``[b x s]``.
            During inference, this indicates the position of the current token.
            This parameter is required during inference if caches have been setup. Default is None.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: output tensor with shape ``[b x s x v]`` or a list of layer
            output tensors defined by ``output_hidden_states`` with the
            final output tensor appended to the list.

    Note:
        At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` should contain the positions of all of the tokens in the prompt.
        For a single-batch prompt, or a batch of prompts with identical lengths, this
        will be ``torch.arange(prompt_length)``. For a batch of varying-length prompts,
        shorter prompts are left-padded and position ids are correspondingly right-shifted,
        thus positional ids should be of shape ``[b, padded_prompt_length]``.
        This is because we will need to retrieve the positional embeddings for each input id.
        In the subsequent steps, if the model has been setup with KV-caches, ``input_pos`` will contain
        the position(s) of the current token(s) ``torch.tensor([padded_prompt_length])``. Otherwise,
        ``input_pos`` will contain all the position ids up to the current token.

    Shape notation:
        - b: batch size
        - s: token sequence length
        - s_e: encoder sequence length
        - v: vocab size
        - d: token embed dim
        - d_e: encoder embed dim
        - m_s: max seq len
    """
    # input tensor of shape [b, s]
    seq_len = tokens.shape[1]

    self._validate_inputs(
        seq_len,
        mask=mask,
        encoder_input=encoder_input,
        encoder_mask=encoder_mask,
        input_pos=input_pos,
    )

    # shape: [b, s, d]
    h = self.tok_embeddings(tokens)

    hidden = []
    for i, layer in enumerate(self.layers):
        if i in self.output_hidden_states:
            hidden.append(h)
        # shape: [b, s, d]
        h = layer(
            h,
            mask=mask,
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )

    h = self.norm(h)

    return h
