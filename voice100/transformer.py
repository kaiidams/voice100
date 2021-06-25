# Copyright 2020 Katsuya Iida. All rights reserved.

import torch
from torch import nn
import numpy as np
import math

_NEG_INF_FP32 = -1e9
_NEG_INF_FP16 = np.finfo(np.float16).min

# Variables

variable_space = ''
trainable_variables = {}

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

def layer_norm(inputs, epsilon=0.001):
    with variable_scope('layer_normalization'):
        mean = torch.mean(inputs, axis=-1)[:, :, None] 
        var = torch.mean((inputs - mean) ** 2, axis=-1)[:, :, None]
        norm_inputs = (inputs - mean) / torch.sqrt(var + epsilon)
        gamma = get_variable('gamma')
        beta = get_variable('beta')
        return gamma * norm_inputs + beta

def dense_layer(name, inputs, subscripts='abc,cde->abde', use_bias=True, activation=None):
    with variable_scope(name):
        kernel = get_variable('kernel')
        y = torch.einsum(subscripts, inputs, kernel)
        if use_bias:
            bias = get_variable('bias')
            y += bias
        if activation is not None:
            y = activation(y)
        return y

# Transformer layers

class EmbeddingSharedWeights(nn.Module):
    __constants__ = ['vocab_size', 'hidden_size']
    voacb_size: int
    hidden_size: int

    def __init__(self, vocab_size, hidden_size):
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.shared_weights = torch.nn.Parameter(torch.Tensor(vocab_size, hidden_size), requires_grad=False)

    def embedding(self, inputs):
        embedded_inputs = self.shared_weights[inputs, :]
        embedded_inputs *= self.hidden_size ** 0.5
        return embedded_inputs

    def linear(self, inputs):
        batch_size = -1 # torch.shape(inputs)[0]
        length = inputs.shape[1]
        x = torch.reshape(inputs, [-1, self.hidden_size])
        logits = torch.matmul(x, self.shared_weights.transpose(0, 1))
        return torch.reshape(logits, [batch_size, length, self.vocab_size])

def get_padding_bias(x: torch.Tensor, padding_value=0, dtype=torch.float32) -> torch.Tensor:
    """
    Args:
        x: tensor of shape [batch_size, length]
    Returns:
        float tensor of shape [batch_size, 1, 1, length]
    """
    assert x.dim() == 2
    neg_inf = _NEG_INF_FP16 if dtype == torch.float16 else _NEG_INF_FP32
    padding = (x == padding_value).to(dtype)
    attention_bias = padding * neg_inf
    attention_bias = attention_bias[:,None,None,:]
    return attention_bias

def get_decoder_self_attention_bias(length, dtype=torch.float32):
    neg_inf = _NEG_INF_FP16 if dtype == torch.float16 else _NEG_INF_FP32
    r = torch.arange(0, length)
    y = (torch.reshape(r, [-1, 1]) < torch.reshape(r, [1, -1])).to(dtype) * neg_inf
    return y[None, None, :, :]

def get_decoder_self_attention_bias_v2(length, dtype=torch.float32):
    """Calculate bias for decoder that maintains model's autoregressive property.

    Creates a tensor that masks out locations that correspond to illegal
    connections, so prediction at position i cannot draw information from future
    positions.

    Args:
        length: int length of sequences in batch.
        dtype: The dtype of the return value.

    Returns:
        float tensor of shape [1, 1, length, length]
    """
    neg_inf = _NEG_INF_FP16 if dtype == torch.float16 else _NEG_INF_FP32
    with torch.name_scope("decoder_self_attention_bias"):
        valid_locs = torch.linalg.band_part(torch.ones([length, length], dtype=dtype),
                                                                         -1, 0)
        valid_locs = torch.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = neg_inf * (1.0 - valid_locs)
    return decoder_bias

def get_position_encoding(
        length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
        length: Sequence length.
        hidden_size: Size of the
        min_timescale: Minimum scale that will be applied at each position
        max_timescale: Maximum scale that will be applied at each position

    Returns:
        Tensor with shape [length, hidden_size]
    """
    # We compute the positional encoding in float32 even if the model uses
    # float16, as many of the ops used, like log and exp, are numerically unstable
    # in float16.
    position = torch.arange(0, length).to(torch.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (num_timescales - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(0, num_timescales).to(torch.float32) * -log_timescale_increment)
    scaled_time = position[:, None] * inv_timescales[None, :]
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=1)
    return signal

def pre_post_processing_wrapper(layer, x, *args):
    with variable_scope("pre_post_processing_wrapper"):
        y = layer_norm(x, epsilon=1e-6)
        y = layer(y, *args)
        return x + y

def self_attention_layer(query_input, bias, name="self_attention", **args):
    return attention_layer(query_input, query_input, bias, name=name, **args)

def attention_layer(query_input, source_input, bias, name="attention", hidden_size=512, num_heads=8):
    with variable_scope(name):
        query = dense_layer('query', query_input, use_bias=False)
        key = dense_layer('key', source_input, use_bias=False)
        value = dense_layer('value', source_input, use_bias=False)

        depth = (hidden_size // num_heads)
        query *= depth ** -0.5

        logits = torch.einsum("btnh,bfnh->bnft", key, query)
        if bias is not None:
            logits += bias
        weights = torch.softmax(logits, dim=3)
        attention_output = torch.einsum("bnft,btnh->bfnh", weights, value)

        attention_output = dense_layer('output_transform', attention_output,
                                         subscripts='abcd,cde->abe',
                                         use_bias=False)
    return attention_output

def feed_forward_network(x):
    with variable_scope("feed_forward_network"):
        output = dense_layer('filter_layer', x, subscripts='abc,cd->abd', activation=torch.relu)
        output = dense_layer('output_layer', output, subscripts='abc,cd->abd')
    return output

# Transformer

def encoder_stack(encoder_inputs, attention_bias, num_layers=6):
    with variable_scope('encoder_stack'):
        for n in range(num_layers):
            with variable_scope("layer_%d" % n):
                with variable_scope("self_attention"):
                    encoder_inputs = pre_post_processing_wrapper(
                        self_attention_layer,
                        encoder_inputs,
                        attention_bias)
                with variable_scope("ffn"):
                    encoder_inputs = pre_post_processing_wrapper(
                        feed_forward_network,
                        encoder_inputs)

        return layer_norm(encoder_inputs, epsilon=1e-6)

def decoder_stack(decoder_inputs,
                    encoder_outputs,
                    decoder_self_attention_bias,
                    attention_bias,
                    num_layers=6):
    with variable_scope('decoder_stack'):
        for n in range(num_layers):
            with variable_scope("layer_%d" % n):
                with variable_scope("self_attention"):
                    decoder_inputs = pre_post_processing_wrapper(
                        self_attention_layer,
                        decoder_inputs,
                        decoder_self_attention_bias)
                with variable_scope("encdec_attention"):
                    decoder_inputs = pre_post_processing_wrapper(
                        attention_layer,
                        decoder_inputs,
                        encoder_outputs,
                        attention_bias)
                with variable_scope("ffn"):
                    decoder_inputs = pre_post_processing_wrapper(
                        feed_forward_network,
                        decoder_inputs)

        return layer_norm(decoder_inputs, epsilon=1e-6)

class TransformerEncoder(nn.Module):
    def forward(self, inputs, embedded_inputs, hidden_size=512):
        with variable_scope('encode'):
            attention_bias = get_padding_bias(inputs, dtype=embedded_inputs.dtype)
            length = embedded_inputs.shape[1]
            pos_encoding = get_position_encoding(length, hidden_size)
            pos_encoding = pos_encoding.to(embedded_inputs.dtype)
            encoder_inputs = embedded_inputs + pos_encoding

            return encoder_stack(encoder_inputs, attention_bias), attention_bias

class TransformerDecoder(nn.Module):
    def forward(self, embedded_targets, encoder_outputs, attention_bias, hidden_size=512):
        with variable_scope("decode"):
            length = embedded_targets.shape[1]
            pos_encoding = get_position_encoding(length, hidden_size)
            pos_encoding = pos_encoding.to(embedded_targets.dtype)
            decoder_inputs = embedded_targets + pos_encoding

            decoder_self_attention_bias = get_decoder_self_attention_bias(length, dtype=embedded_targets.dtype)
            outputs = decoder_stack(
                decoder_inputs,
                encoder_outputs,
                decoder_self_attention_bias,
                attention_bias)

        return outputs

class Transformer(nn.Module):
    __constants__ = ['vocab_size', 'hidden_size']

    def __init__(self, vocab_size: int, hidden_size: int, arr=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dtype = torch.float32
        self.encode = TransformerEncoder()
        self.decode = TransformerDecoder()
        self.embedding_softmax_layer = EmbeddingSharedWeights(self.vocab_size, self.hidden_size)
        if arr is not None:
            set_variables(arr)
            for k, v in trainable_variables.items():
                if k != 'encode/embedding_shared_weights/embedding_and_softmax/weights':
                    setattr(self, k.replace('/', '_'), v)
        with torch.no_grad():
            with variable_scope('encode/embedding_shared_weights/embedding_and_softmax'):
                self.embedding_softmax_layer.shared_weights.copy_(get_variable('weights').detach())

    def forward(self, inputs, targets):
        embedded_inputs = self.embedding_softmax_layer.embedding(inputs)
        embedded_targets = self.embedding_softmax_layer.embedding(targets)
        print(embedded_inputs.shape, embedded_inputs.shape)
        encoder_outputs, attention_bias = self.encode(inputs, embedded_inputs)
        decoder_outputs = self.decode(embedded_targets, encoder_outputs, attention_bias)
        logits = self.embedding_softmax_layer.linear(decoder_outputs)
        outputs = logits[:, -1:, :].argmax(dim=2).to(torch.long)
        return logits, outputs

class InferTransformer(Transformer):
    __constants__ = ['vocab_size', 'hidden_size']

    def __init__(self, arr=None):
        super(InferTransformer, self).__init__(arr)
        self.vocab_size = 64003
        self.hidden_size = 512
        self.encode = torch.jit.trace(
            TransformerEncoder(),
            (torch.ones(size=(1, 1), dtype=torch.long),
            torch.ones(size=(1, 1, 512), dtype=self.dtype)))
        self.decode = torch.jit.trace(
            TransformerDecoder(),
            (torch.ones(size=(1, 1, 512), dtype=self.dtype),
            torch.ones(size=(1, 1, 512), dtype=self.dtype),
            torch.ones(size=(1, 1, 1), dtype=self.dtype)))
        self.embedding_softmax_layer = torch.jit.trace_module(
            EmbeddingSharedWeights(self.vocab_size, self.hidden_size),
            {
                'embedding': torch.ones(size=(1, 1), dtype=torch.long),
                'linear': torch.ones(size=(1, 1, 512), dtype=torch.float32),
            })
        with torch.no_grad():
            with variable_scope('encode/embedding_shared_weights/embedding_and_softmax'):
                self.embedding_softmax_layer.shared_weights.copy_(get_variable('weights').detach())

    def forward(self, inputs, targets):
        embedded_inputs = self.embedding_softmax_layer.embedding(inputs)
        encoder_outputs, attention_bias = self.encode(inputs, embedded_inputs)
        while (
            (targets.shape[1] < torch.tensor(20, dtype=torch.long)).to(torch.long) *
            (targets[0, -1] != torch.tensor(2, dtype=torch.long)).to(torch.long)):
            embedded_targets = self.embedding_softmax_layer.embedding(targets)
            decoder_outputs = self.decode(embedded_targets, encoder_outputs, attention_bias)
            logits = self.embedding_softmax_layer.linear(decoder_outputs)

            outputs = logits[:, -1:, :].argmax(dim=2).to(torch.long)
            targets = torch.cat([
                targets, outputs
            ], dim=1)
        return targets

def load_model(file):
    arr = np.load(file)
    return Transformer(64003, 512, arr)

def load_model2(file):
    arr = np.load(file)
    return InferTransformer(arr)
