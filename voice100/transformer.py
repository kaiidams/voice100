# Copyright 2020 Katsuya Iida. All rights reserved.

import torch
from torch import nn
import numpy as np
import math

from torch.nn.modules.normalization import LayerNorm

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

def layer_norm(inputs, epsilon=0.001):
    with variable_scope('layer_normalization'):
        mean = torch.mean(inputs, axis=-1)[:, :, None] 
        var = torch.mean((inputs - mean) ** 2, axis=-1)[:, :, None]
        norm_inputs = (inputs - mean) / torch.sqrt(var + epsilon)
        gamma = get_variable('gamma')
        beta = get_variable('beta')
        return gamma * norm_inputs + beta

def load_numpy_state_layer_norm(layer):
    with variable_scope('layer_normalization'):
        layer.weight[:] = get_variable('gamma')
        layer.bias[:] = get_variable('beta')

class EinSumLinear(nn.Module):
    def __init__(self, subscripts: str, weight_shape, bias_shape=None,
        device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.subscripts = subscripts
        self.weight = nn.Parameter(torch.empty(weight_shape, **factory_kwargs))
        if bias_shape is not None:
            self.bias = nn.Parameter(torch.empty(bias_shape, **factory_kwargs))
        else:
            self.bias = None
        #self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = torch.einsum(self.subscripts, input, self.weight)
        if self.bias is not None:
            y += self.bias
        return y

def load_numpy_state_dense_layer(name, layer):
    with variable_scope(name):
        layer.weight[:] = get_variable('kernel')
        if layer.bias is not None:
            layer.bias[:] = get_variable('bias')

def dense_layer(name, inputs, subscripts='abc,cde->abde', use_bias=True, activation=None):
    with variable_scope(name):
        kernel = get_variable('kernel')
        y = torch.einsum(subscripts, inputs, kernel)
        if use_bias:
            bias = get_variable('bias')
            y += bias
        if False:
            if not has_current_module():
                if use_bias:
                    kernel = get_variable('kernel')
                    bias = get_variable('bias')
                    linear = EinSumLinear(subscripts=subscripts, weight_shape=kernel.shape, bias_shape=bias.shape)
                    linear.weight = kernel
                    linear.weight = bias
                else:
                    kernel = get_variable('kernel')
                    linear = EinSumLinear(subscripts=subscripts, weight_shape=kernel.shape)
                    linear.weight = kernel
                set_current_module(linear)
            print(current_module())
            linear = current_module()
            y = linear(inputs)
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

class PrePostProcessingWrapper(nn.Module):

    def __init__(self, layer, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layer = layer

    def load_numpy_state(self):
        with variable_scope("pre_post_processing_wrapper"):
            load_numpy_state_layer_norm(self.layer_norm)
            if isinstance(self.layer, nn.Module):
                self.layer.load_numpy_state()

    def forward(self, x, *args):
        with variable_scope("pre_post_processing_wrapper"):
            y = layer_norm(x, epsilon=1e-6)
            y = self.layer(y, *args)
            return x + y

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

class FeedForwardNetwork(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.filter_layer = EinSumLinear(
            subscripts='abc,cd->abd',
            weight_shape=[hidden_size, 2048],
            bias_shape=[2048])
        self.activation = nn.ReLU()
        self.output_layer = EinSumLinear(
            subscripts='abc,cd->abd',
            weight_shape=[2048, hidden_size],
            bias_shape=[hidden_size])

    def load_numpy_state(self):
        with variable_scope("feed_forward_network"):
            load_numpy_state_dense_layer('filter_layer', self.filter_layer)
            load_numpy_state_dense_layer('output_layer', self.output_layer)

    def forward(self, input):
        with variable_scope("feed_forward_network"):
            x = self.filter_layer(input)
            x = self.activation(x)
            x = self.output_layer(x)
            return x

def feed_forward_network(x):
    with variable_scope("feed_forward_network"):
        output = dense_layer('filter_layer', x, subscripts='abc,cd->abd', activation=torch.relu)
        output = dense_layer('output_layer', output, subscripts='abc,cd->abd')
    return output

# Transformer

class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attention = PrePostProcessingWrapper(self_attention_layer, hidden_size)
        self.ffn = PrePostProcessingWrapper(FeedForwardNetwork(hidden_size), hidden_size)

    def load_numpy_state(self):
        with variable_scope("self_attention"):
            self.self_attention.load_numpy_state()
        with variable_scope("ffn"):
            self.ffn.load_numpy_state()

    def forward(self, inputs, attention_bias):
        with variable_scope("self_attention"):
            x = self.self_attention(inputs, attention_bias)
        with variable_scope("ffn"):
            x = self.ffn(x)
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        layers = []
        for i in range(self.num_layers):
            layers.append(TransformerEncoderLayer(hidden_size=hidden_size))
        layers.append(nn.LayerNorm(hidden_size, eps=1e-6))
        self.layers = nn.Sequential(*layers)

    def load_numpy_state(self):
        with variable_scope("encode"):
            with variable_scope('encoder_stack'):
                for n in range(self.num_layers):
                    with variable_scope("layer_%d" % n):
                        self.layers[n].load_numpy_state()

                load_numpy_state_layer_norm(self.layers[self.num_layers])

    def forward(self, inputs, embedded_inputs):
        with variable_scope('encode'):
            attention_bias = get_padding_bias(inputs, dtype=embedded_inputs.dtype)
            length = embedded_inputs.shape[1]
            pos_encoding = get_position_encoding(length, self.hidden_size)
            pos_encoding = pos_encoding.to(embedded_inputs.dtype)
            encoder_inputs = embedded_inputs + pos_encoding

            return self.encoder_stack(encoder_inputs, attention_bias), attention_bias

    def encoder_stack(self, encoder_inputs, attention_bias):
        with variable_scope('encoder_stack'):
            for n in range(self.num_layers):
                with variable_scope("layer_%d" % n):
                    encoder_inputs = self.layers[n](encoder_inputs, attention_bias)

        return self.layers[self.num_layers](encoder_inputs)

class TransformerDecoder(nn.Module):

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, embedded_targets, encoder_outputs, attention_bias):
        with variable_scope("decode"):
            length = embedded_targets.shape[1]
            pos_encoding = get_position_encoding(length, self.hidden_size)
            pos_encoding = pos_encoding.to(embedded_targets.dtype)
            decoder_inputs = embedded_targets + pos_encoding

            decoder_self_attention_bias = get_decoder_self_attention_bias(length, dtype=embedded_targets.dtype)
            outputs = self.decoder_stack(
                decoder_inputs,
                encoder_outputs,
                decoder_self_attention_bias,
                attention_bias)

        return outputs

    def decoder_stack(
        self, decoder_inputs, encoder_outputs,
        decoder_self_attention_bias, attention_bias
        ):
        with variable_scope('decoder_stack'):
            for n in range(self.num_layers):
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

class Transformer(nn.Module):
    __constants__ = ['vocab_size', 'hidden_size']

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, arr=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dtype = torch.float32
        self.encode = TransformerEncoder(hidden_size=hidden_size, num_layers=num_layers)
        self.decode = TransformerDecoder(hidden_size=hidden_size, num_layers=num_layers)
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

    def load_numpy_state(self):
        self.encode.load_numpy_state()

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
    return Transformer(64003, 512, 6, arr)

def load_model2(file):
    arr = np.load(file)
    return InferTransformer(arr)

def main():
    model_file = '/home/kaiida/data/brokenegg/brokenegg.npz'
    vocab_file = '/home/kaiida/data/brokenegg/brokenegg.en-es-ja.spm64k.model'
    model = load_model(model_file)
    for n in model.state_dict().keys():
        print(n)
    with torch.no_grad():
        model.load_numpy_state()
    inputs = torch.tensor([[  393,  1244,  1268, 21851,    37,     8,  1174, 12024,  1396, 22667,
            157,   116,  1389,    11,  5662, 13199,    45, 27204,    19,  3811,
             16,  3369, 18380, 34191,     3,     1,     0,     0,     0]], dtype=torch.long)
    targets = torch.tensor([[64002,     6, 32588, 31560,    20,  1461, 10160, 10971,    28,  3361,
          2889,  1461]], dtype=torch.long)
    with torch.no_grad():
        logits, outputs = model(inputs, targets)
        targets = torch.cat([targets, outputs], axis=1)
    print(outputs)
    print(targets)
    logits_ = torch.load('/home/kaiida/data/brokenegg/brokenegg_test.pt')
    mse = torch.square(logits_ - logits).mean().item()
    print(mse)
    print(targets[0, -1] == 10160)

main()