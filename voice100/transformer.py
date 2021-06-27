# Copyright 2020 Katsuya Iida. All rights reserved.

import torch
from torch import nn
from torch.nn import init
import numpy as np
import math

__all__ = ["Transformer", "Translation"]

_NEG_INF_FP32 = -1e9
_NEG_INF_FP16 = np.finfo(np.float16).min

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
def get_padding_bias(x: torch.Tensor, length: torch.Tensor, padding_value=0) -> torch.Tensor:
    """
    Args:
        x: tensor of shape [batch_size, length, embed_size]
        length: tensor of shape [batch_size]
    Returns:
        float tensor of shape [batch_size, 1, 1, length]
    """
    assert x.dim() == 3
    assert length.dim() == 1
    neg_inf = _NEG_INF_FP16 if x.dtype == torch.float16 else _NEG_INF_FP32
    padding = (torch.arange(x.shape[1], device=x.device)[None, :] >= length[:, None]).float()
    attention_bias = padding * neg_inf
    attention_bias = attention_bias[:,None,None,:]
    return attention_bias

@torch.no_grad()
def get_decoder_self_attention_bias(length, device, dtype=torch.float32):
    neg_inf = _NEG_INF_FP16 if dtype == torch.float16 else _NEG_INF_FP32
    r = torch.arange(0, length, device=device)
    y = (torch.reshape(r, [-1, 1]) < torch.reshape(r, [1, -1])).to(dtype) * neg_inf
    return y[None, None, :, :]

@torch.no_grad()
def get_position_encoding(
        length, hidden_size, device, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Returns:
        Tensor with shape [length, hidden_size]
    """
    position = torch.arange(0, length, device=device, dtype=torch.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (num_timescales - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(0, num_timescales, device=device, dtype=torch.float32) * -log_timescale_increment)
    scaled_time = position[:, None] * inv_timescales[None, :]
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=1)
    return signal

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

    def load_numpy_state(self):
        with variable_scope('attention'):
            x = get_variable('query/kernel')
            #print(x.shape)
            self.layer.in_proj_weight[:512, :] = get_variable('query/kernel').reshape([512, 512]).T
            self.layer.in_proj_weight[512:1024, :] = get_variable('key/kernel').reshape([512, 512]).T
            self.layer.in_proj_weight[1024:, :] = get_variable('value/kernel').reshape([512, 512]).T
            x = get_variable('output_transform/kernel')
            #print(x.shape)
            self.layer.out_proj.weight[:, :] = get_variable('output_transform/kernel').reshape([512, 512]).T

    def forward(self, query_input, source_input, bias):
        x = bias[:, 0, 0, :] == 0
        #print(x)
        #print(bias.shape)
        x, _ = self.layer(query_input, source_input, source_input, key_padding_mask=bias[:, 0, 0, :] != 0)
        return x

class SelfAttentionLayer(AttentionLayer):
    def load_numpy_state(self):
        with variable_scope('self_attention'):
            self.layer.in_proj_weight[:512, :] = get_variable('query/kernel').reshape([512, 512]).T
            self.layer.in_proj_weight[512:1024, :] = get_variable('key/kernel').reshape([512, 512]).T
            self.layer.in_proj_weight[1024:, :] = get_variable('value/kernel').reshape([512, 512]).T
            self.layer.out_proj.weight[:, :] = get_variable('output_transform/kernel').reshape([512, 512]).T

    def forward(self, query_input, bias, **args):
        #print(bias.shape)
        if bias.shape[2] != 1:
            x = bias[0, 0, :, :] != 0
            #print(x.int())
            x, _ = self.layer(query_input, query_input, query_input, need_weights=False, attn_mask=bias[0, 0, :, :] != 0)
        else:
            x = bias[:, 0, 0, :] == 0
            #print(x)
            x, _ = self.layer(query_input, query_input, query_input, need_weights=False, key_padding_mask=bias[:, 0, 0, :] != 0)
        #print(x.shape)
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
        self.self_attention = PrePostProcessingWrapper(SelfAttentionLayer(hidden_size, num_heads), hidden_size)
        self.ffn = PrePostProcessingWrapper(FeedForwardNetwork(hidden_size, filter_size), hidden_size)

    def load_numpy_state(self):
        with variable_scope("self_attention"):
            self.self_attention.load_numpy_state()
        with variable_scope("ffn"):
            self.ffn.load_numpy_state()

    def forward(self, inputs, attention_bias):
        x = self.self_attention(inputs, attention_bias)
        x = self.ffn(x)
        return x

class TransformerEncoder(nn.Module):

    def __init__(
        self, num_layers: int, hidden_size: int, filter_size: int, num_heads: int, dropout: float = 0.1,
        device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        layers = []
        for i in range(self.num_layers):
            layers.append(TransformerEncoderLayer(hidden_size=hidden_size, filter_size=filter_size, num_heads=num_heads))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6, **factory_kwargs)

    def load_numpy_state(self):
        with variable_scope("encode"):
            with variable_scope('encoder_stack'):
                for n in range(self.num_layers):
                    with variable_scope("layer_%d" % n):
                        self.layers[n].load_numpy_state()

                load_numpy_state_layer_norm(self.layer_norm)

    def forward(self, embedded_inputs, attention_bias):
        length = embedded_inputs.shape[1]
        pos_encoding = get_position_encoding(length, self.hidden_size, device=embedded_inputs.device)
        pos_encoding = pos_encoding.to(embedded_inputs.dtype)
        encoder_inputs = self.dropout(embedded_inputs + pos_encoding)
        return self.encoder_stack(encoder_inputs, attention_bias)

    def encoder_stack(self, encoder_inputs, attention_bias):
        for layer in self.layers:
            encoder_inputs = layer(encoder_inputs, attention_bias)
        return self.layer_norm(encoder_inputs)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, hidden_size: int, filter_size: int, num_heads: int,
        device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attention = PrePostProcessingWrapper(SelfAttentionLayer(hidden_size, num_heads), hidden_size)
        self.encdec_attention = PrePostProcessingWrapper(AttentionLayer(hidden_size, num_heads), hidden_size)
        self.ffn = PrePostProcessingWrapper(FeedForwardNetwork(hidden_size, filter_size), hidden_size)

    def load_numpy_state(self):
        with variable_scope("self_attention"):
            self.self_attention.load_numpy_state()
        with variable_scope("encdec_attention"):
            self.encdec_attention.load_numpy_state()
        with variable_scope("ffn"):
            self.ffn.load_numpy_state()

    def forward(self, decoder_inputs, encoder_outputs,
        decoder_self_attention_bias, attention_bias):
        x = self.self_attention(decoder_inputs, decoder_self_attention_bias)
        x = self.encdec_attention(x, encoder_outputs, attention_bias)
        x = self.ffn(x)
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, num_layers: int, hidden_size: int, filter_size: int, num_heads: int, dropout: float = 0.1,
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
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6, **factory_kwargs)

    def load_numpy_state(self):
        with variable_scope("decode"):
            with variable_scope('decoder_stack'):
                for n in range(self.num_layers):
                    with variable_scope("layer_%d" % n):
                        self.layers[n].load_numpy_state()

                load_numpy_state_layer_norm(self.layer_norm)

    def forward(self, embedded_targets, encoder_outputs, attention_bias):
        length = embedded_targets.shape[1]
        pos_encoding = get_position_encoding(length, self.hidden_size, device=embedded_targets.device)
        pos_encoding = pos_encoding.to(embedded_targets.dtype)
        decoder_inputs = self.dropout(embedded_targets + pos_encoding)

        decoder_self_attention_bias = get_decoder_self_attention_bias(
            length, device=embedded_targets.device, dtype=embedded_targets.dtype)
        return self.decoder_stack(
            decoder_inputs,
            encoder_outputs,
            decoder_self_attention_bias,
            attention_bias)

    def decoder_stack(
        self, decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias
        ):
        x = decoder_inputs
        for layer in self.layers:
            x = layer(x, encoder_outputs, decoder_self_attention_bias, attention_bias)
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

    def forward(self, embedded_inputs, inputs_len, embedded_targets):
        attention_bias = get_padding_bias(embedded_inputs, inputs_len)
        encoder_outputs = self.encode(embedded_inputs, attention_bias)
        decoder_outputs = self.decode(embedded_targets, encoder_outputs, attention_bias)
        return decoder_outputs

class Translation(nn.Module):
    __constants__ = ['vocab_size', 'hidden_size']

    def __init__(self, vocab_size: int, hidden_size: int, filter_size: int,
        num_layers: int, num_heads: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            dim_feedforward=filter_size,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            nhead=num_heads,
            batch_first=True,
            **factory_kwargs)
        self.embedding = nn.Embedding(
            vocab_size, hidden_size, **factory_kwargs)

    def load_numpy_state(self):
        self.transformer.load_numpy_state()
        with variable_scope('encode/embedding_shared_weights/embedding_and_softmax'):
            self.embedding.weight[:] = get_variable('weights')

    def forward(self, inputs, inputs_len, targets):
        embedded_inputs = self.embedding(inputs) * self.hidden_size ** 0.5
        embedded_targets = self.embedding(targets) * self.hidden_size ** 0.5
        src_key_padding_mask = get_padding_bias(embedded_inputs, inputs_len)
        decoder_self_attention_bias = get_decoder_self_attention_bias(
            embedded_targets.shape[1], device=embedded_targets.device, dtype=embedded_targets.dtype)
        decoder_outputs = self.transformer(embedded_inputs, embedded_targets,
            tgt_mask=decoder_self_attention_bias[0, 0, :, :],
            src_key_padding_mask=src_key_padding_mask[:, 0, 0, :])

        batch_size = -1 # torch.shape(inputs)[0]
        length = decoder_outputs.shape[1]
        x = torch.reshape(decoder_outputs, [-1, self.hidden_size])
        logits = torch.matmul(x, self.embedding.weight.transpose(0, 1))
        return torch.reshape(logits, [batch_size, length, self.vocab_size])

def load_model(file, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    arr = np.load(file)
    set_variables(arr)
    transformer = Translation(64003, 512, 2048, 6, 8, **factory_kwargs)
    return transformer