# Copyright 2020 Katsuya Iida. All rights reserved.

import torch
from torch import nn
import numpy as np
import math
from typing import Dict

__all__ = ["Transformer", "Translation"]

# Variables

variable_space = ''
trainable_variables = {}
regsitered_modules = {}

class _VariableScope:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        global variable_space
        self.parent = variable_space
        if variable_space:
            variable_space += '/' + self.name
        else:
            variable_space = self.name
    def __exit__(self, a, b, c):
        global variable_space
        variable_space = self.parent

def variable_scope(name):
    return _VariableScope(name)

def has_current_module():
    return variable_space in regsitered_modules

def set_current_module(m):
    regsitered_modules[variable_space] = m

def current_module():
    return regsitered_modules[variable_space]

def get_variable(name):
    return trainable_variables[variable_space + '/' + name]

def set_variable(name, value):
    trainable_variables[variable_space + '/' + name] = value

def set_variables(vars):
    global trainable_variables
    trainable_variables = {
        k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False)
        for k, v in vars.items()
    }

def list_variables():
    return [
        (k, v.shape)
        for k, v in trainable_variables.items()
    ]

# Misc

def load_numpy_state_layer_norm(layer):
    with variable_scope('layer_normalization'):
        layer.weight.copy_(get_variable('gamma'))
        layer.bias.copy_(get_variable('beta'))

# Transformer layers

@torch.no_grad()
def generate_key_padding_mask(x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: tensor of shape [batch_size, length, embed_size]
        length: tensor of shape [batch_size]
    Returns:
        float tensor of shape [batch_size, length]
    """
    assert x.dim() == 2 or x.dim() == 3
    assert length.dim() == 1
    return torch.arange(x.shape[1], device=x.device)[None, :] >= length[:, None]

@torch.no_grad()
def generate_square_subsequent_mask(sz: int, device) -> torch.Tensor:
    """
    Args:
        sz: size of the mask
    Returns:
        float tensor of shape [sz, sz]
    """
    r = torch.arange(0, sz, device=device)
    return r[:, None] < r[None, :]

@torch.no_grad()
def generate_position_encoding(
    x: torch.Tensor, min_timescale=1.0, max_timescale=1.0e4
    ) -> torch.Tensor:
    """Return positional encoding.

    Args:
        x: tensor of shape [batch_size, length, hidden_size]
    Returns:
        Tensor with shape [length, hidden_size]
    """
    assert x.dim() == 3
    length = x.shape[1]
    hidden_size = x.shape[2]
    position = torch.arange(0, length, device=x.device, dtype=torch.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (num_timescales - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(0, num_timescales, device=x.device, dtype=torch.float32) * -log_timescale_increment)
    scaled_time = position[:, None] * inv_timescales[None, :]
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=1)
    return signal.to(x.dtype)

class PrePostProcessingWrapper(nn.Module):

    def __init__(self, layer, hidden_size: int, dropout: float = 0.1,
        device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6, **factory_kwargs)
        self.layer = layer
        self.dropout = nn.Dropout(dropout, inplace=True)

    def load_numpy_state(self):
        with variable_scope("pre_post_processing_wrapper"):
            load_numpy_state_layer_norm(self.layer_norm)
            if isinstance(self.layer, nn.Module):
                self.layer.load_numpy_state()

    def forward(self, x, *args, **kwargs):
        y = self.layer_norm(x)
        y = self.layer(y, *args, **kwargs)
        y = self.dropout(y)
        return x + y

class AttentionLayer(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.layer = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout,
            bias=False, batch_first=True, **factory_kwargs)

    def load_numpy_state(self, name='attention'):
        with variable_scope(name):
            self.layer.in_proj_weight[:512, :] = get_variable('query/kernel').reshape([512, 512]).T
            self.layer.in_proj_weight[512:1024, :] = get_variable('key/kernel').reshape([512, 512]).T
            self.layer.in_proj_weight[1024:, :] = get_variable('value/kernel').reshape([512, 512]).T
            self.layer.out_proj.weight[:, :] = get_variable('output_transform/kernel').reshape([512, 512]).T

    def forward(self, query_input, source_input, key_padding_mask):
        x, _ = self.layer(query_input, source_input, source_input, key_padding_mask=key_padding_mask)
        return x

class SelfAttentionLayer(AttentionLayer):
    def load_numpy_state(self, name='self_attention'):
        super(self).load_numpy_state(name)

    def forward(
        self, query_input, key_padding_mask=None, attn_mask=None,
        cache_key: str = None, cache: Dict[str, torch.Tensor] = None
        ) -> torch.Tensor:
        if cache is not None:
            if cache_key in cache:
                y = torch.cat([cache[cache_key], query_input], axis=1)
            else:
                y = query_input
            cache[cache_key] = y
            x, _ = self.layer(
                query_input, y, y,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            x, _ = self.layer(
                query_input, query_input, query_input,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return x

class FeedForwardNetwork(nn.Sequential):

    def __init__(self, hidden_size, filter_size,
        device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            nn.Linear(hidden_size, filter_size, bias=True, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(filter_size, hidden_size, bias=True, **factory_kwargs))

    def load_numpy_state(self):
        with variable_scope("feed_forward_network"):
            with variable_scope("filter_layer"):
                self[0].weight[:] = get_variable('kernel').T
                self[0].bias[:] = get_variable('bias').T
            with variable_scope("output_layer"):
                self[2].weight[:] = get_variable('kernel').T
                self[2].bias[:] = get_variable('bias').T

# Transformer

class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size: int, filter_size: int, num_heads: int,
        device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attention = PrePostProcessingWrapper(SelfAttentionLayer(hidden_size, num_heads, **factory_kwargs), hidden_size)
        self.ffn = PrePostProcessingWrapper(FeedForwardNetwork(hidden_size, filter_size, **factory_kwargs), hidden_size)

    def load_numpy_state(self):
        with variable_scope("self_attention"):
            self.self_attention.load_numpy_state()
        with variable_scope("ffn"):
            self.ffn.load_numpy_state()

    def forward(self, inputs, src_key_padding_mask):
        x = self.self_attention(inputs, key_padding_mask=src_key_padding_mask)
        x = self.ffn(x)
        return x

class TransformerEncoder(nn.Module):

    def __init__(
        self, num_layers: int, hidden_size: int, filter_size: int, num_heads: int,
        device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        layers = []
        for i in range(self.num_layers):
            layers.append(TransformerEncoderLayer(hidden_size=hidden_size, filter_size=filter_size, num_heads=num_heads))
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6, **factory_kwargs)

    def load_numpy_state(self):
        with variable_scope("encode"):
            with variable_scope('encoder_stack'):
                for n in range(self.num_layers):
                    with variable_scope("layer_%d" % n):
                        self.layers[n].load_numpy_state()

                load_numpy_state_layer_norm(self.layer_norm)

    def forward(self, src, src_key_padding_mask):
        x = src
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return self.layer_norm(x)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, hidden_size: int, filter_size: int, num_heads: int,
        device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attention = PrePostProcessingWrapper(SelfAttentionLayer(hidden_size, num_heads, **factory_kwargs), hidden_size)
        self.encdec_attention = PrePostProcessingWrapper(AttentionLayer(hidden_size, num_heads, **factory_kwargs), hidden_size)
        self.ffn = PrePostProcessingWrapper(FeedForwardNetwork(hidden_size, filter_size, **factory_kwargs), hidden_size)

    def load_numpy_state(self):
        with variable_scope("self_attention"):
            self.self_attention.load_numpy_state()
        with variable_scope("encdec_attention"):
            self.encdec_attention.load_numpy_state()
        with variable_scope("ffn"):
            self.ffn.load_numpy_state()

    def forward(
        self, decoder_inputs, encoder_outputs,
        tgt_mask, memory_key_padding_mask,
        cache_key: str, cache: Dict[str, torch.Tensor] = None
        ) -> torch.Tensor:
        x = self.self_attention(
            decoder_inputs,
            attn_mask=tgt_mask,
            cache_key=cache_key,
            cache=cache)
        x = self.encdec_attention(x, encoder_outputs, key_padding_mask=memory_key_padding_mask)
        x = self.ffn(x)
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, num_layers: int, hidden_size: int, filter_size: int, num_heads: int,
        device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                TransformerDecoderLayer(hidden_size=hidden_size, filter_size=filter_size, num_heads=num_heads,
                    **factory_kwargs))
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6, **factory_kwargs)

    def load_numpy_state(self):
        with variable_scope("decode"):
            with variable_scope('decoder_stack'):
                for n in range(self.num_layers):
                    with variable_scope("layer_%d" % n):
                        self.layers[n].load_numpy_state()

                load_numpy_state_layer_norm(self.layer_norm)

    def forward(
        self, tgt, src, tgt_mask, memory_key_padding_mask,
        cache: Dict[str, torch.Tensor] = None
        ):
        x = tgt
        for i, layer in enumerate(self.layers):
            x = layer(
                x, src, tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                cache_key='dec%d' % i,
                cache=cache)
        return self.layer_norm(x)

class Transformer(nn.Module):
    __constants__ = ['hidden_size']

    def __init__(
        self, hidden_size: int, filter_size: int,
        num_layers: int, num_heads: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.encode = TransformerEncoder(
            hidden_size=hidden_size, filter_size=filter_size,
            num_layers=num_layers, num_heads=num_heads, **factory_kwargs)
        self.decode = TransformerDecoder(
            hidden_size=hidden_size, filter_size=filter_size,
            num_layers=num_layers, num_heads=num_heads, **factory_kwargs)

    def load_numpy_state(self):
        self.encode.load_numpy_state()
        self.decode.load_numpy_state()

    def forward(
        self, src, tgt, src_key_padding_mask, tgt_mask
        ) -> torch.Tensor:
        encoder_outputs = self.encode(
            src, src_key_padding_mask)
        decoder_outputs = self.decode(
            tgt, encoder_outputs,
            tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
        return decoder_outputs

class Translation(nn.Module):
    __constants__ = ['vocab_size', 'hidden_size']

    def __init__(self, native: bool, vocab_size: int, hidden_size: int, filter_size: int,
        num_layers: int, num_heads: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.native = native
        if self.native:
            self.transformer = nn.Transformer(
                d_model=hidden_size,
                dim_feedforward=filter_size,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                nhead=num_heads,
                batch_first=True,
                **factory_kwargs)
        else:
            self.transformer = Transformer(
                hidden_size=hidden_size,
                filter_size=filter_size,
                num_layers=num_layers,
                num_heads=num_heads,
                **factory_kwargs)
        self.embedding = nn.Embedding(
            vocab_size, hidden_size, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        a = 1 / math.sqrt(self.hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0, std=a)

    def load_numpy_state(self):
        self.transformer.load_numpy_state()
        with variable_scope('encode/embedding_shared_weights/embedding_and_softmax'):
            self.embedding.weight[:] = get_variable('weights')

    def forward(self, inputs, inputs_len, targets):
        embedded_inputs = self.embedding(inputs) * self.hidden_size ** 0.5
        embedded_targets = self.embedding(targets) * self.hidden_size ** 0.5
        if self.native:
            src_key_padding_mask = generate_key_padding_mask(embedded_inputs, inputs_len)
            tgt_mask = generate_square_subsequent_mask(
                embedded_targets.shape[1], device=embedded_targets.device, dtype=embedded_targets.dtype)
            decoder_outputs = self.transformer(embedded_inputs, embedded_targets,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask)
        else:
            decoder_outputs = self.transformer(embedded_inputs, inputs_len, embedded_targets)

        batch_size = -1 # torch.shape(inputs)[0]
        length = decoder_outputs.shape[1]
        x = torch.reshape(decoder_outputs, [-1, self.hidden_size])
        logits = torch.matmul(x, self.embedding.weight.transpose(0, 1))
        return (
            torch.reshape(logits, [batch_size, length, self.vocab_size]),
            torch.reshape(decoder_outputs, [batch_size, length, self.hidden_size]))

def load_model(file, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    arr = np.load(file)
    set_variables(arr)
    transformer = Translation(64003, 512, 2048, 6, 8, **factory_kwargs)
    return transformer