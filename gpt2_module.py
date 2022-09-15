import torch 
import torch.nn as nn 
import math
import sys
import psutil
import os 
from argparse import ArgumentParser, REMAINDER
import intel_extension_for_pytorch as ipex
import time 
class Config:
     def __init__(self):
         self.hidden_size = 6144
         self.num_hidden_layers = 44
         self.vocab_size = 50432
         self.pad_token_id = 0
         self.layer_norm_eps = 0.00001
         self.hidden_dropout_prob = 0.5
         self.num_attention_heads = 64
         self.seq_len = 2048
         self.attention_probs_dropout_prob= 0.0005

config = Config()         
	
class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self,
        input_ids):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask = None):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)

        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = SelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    
    def forward(
        self,
        hidden_states,
        attention_mask = None):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size*4)
        self.intermediate_act_fn = torch.nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*4, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.hidden_size*4
        self.seq_len_dim = 1
        self.attention = Attention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )

        layer_output = self.intermediate(self_attention_outputs)
        layer_output = self.output(layer_output,hidden_states)


        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class GPT2Module(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.layer = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)])
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)
    def forward(
        self,
        input_ids,
        attention_mask = None,
    ):
        hidden_states = self.embeddings(input_ids)
        for i, layer_module in enumerate(self.layer):
            layer_output = layer_module(
                    hidden_states,
                    attention_mask,
                )

            hidden_states = layer_output
        hidden_states = self.linear(hidden_states)
        return hidden_states 

parser = ArgumentParser()
parser.add_argument("--fp32", action='store_true', default=False,
                        help="Enable FP32")
args=parser.parse_args()

loss_fn = nn.MSELoss()
process = psutil.Process(os.getpid())
print("Memory usage:", process.memory_info().rss/1024/1024/1024, "GB", flush=True)
model = GPT2Module(config)
print("Memory usage before ipex.optimize:", process.memory_info().rss/1024/1024/1024, "GB", flush=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16 if not args.fp32 else torch.float, inplace=True)
print("Memory usage after  ipex.optimize:", process.memory_info().rss/1024/1024/1024, "GB", flush=True)
input = torch.clamp(torch.randn(2,config.seq_len), min=1, max=config.vocab_size-1).to(torch.int32)

for i in range(1000):
    print("Parameter memory usage:", process.memory_info().rss/1024/1024/1024, "GB", flush=True)
    start = time.time()
    with torch.cpu.amp.autocast(enabled=not args.fp32):
        res = model(input)
        loss =  loss_fn(res,res)
        print("Memory usage after forward and before backward:", process.memory_info().rss/1024/1024/1024, "GB", flush=True)
        loss.backward()
        print("Memory usage after backward and before optimizer step:", process.memory_info().rss/1024/1024/1024, "GB", flush=True)    
    optimizer.step()
    end = time.time()
    print("********Iteration {}: {} ms/it, dtype= {}**********".format(i, 1000*(end-start), 'fp32' if args.fp32 else 'bf16'))


