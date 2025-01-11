import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_length, dropout,
                 num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # 1

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out) # 2

        self.dropout = nn.Dropout(dropout) 
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        ) 

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # 3
        queries = self.W_query(x) # 3
        values = self.W_value(x) # 3

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # 4
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2) # 5
        queries = queries.transpose(1, 2) # 5
        values = values.transpose(1, 2) # 5

        attn_scores = queries @ keys.transpose(2, 3) # omega # 6
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # 7

        attn_scores.masked_fill_(mask_bool, -torch.inf) # 8

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) # 9
        context_vec = context_vec.contiguous().view( # 10
            b, num_tokens, self.d_out
        )

        context_vec = self.out_proj(context_vec) # 11
        return context_vec


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # layers to train the model
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
                (x + 0.044715 * torch.pow(x, 3))
            )
        )

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # multi-head attention
        self.att = MultiHeadAttention(
            # input dim
            d_in = cfg["emb_dim"],
            # output dim
            d_out = cfg["emb_dim"],
            # actual input length
            context_length = cfg["context_length"],
            # number of causal attention 
            num_heads = cfg["n_heads"],
            # masking rate
            dropout = cfg["drop_rate"],
            # if adding query, key, and value bias
            qkv_bias = cfg["qkv_bias"]
        )

        # Apply layers and activation function to train the model
        self.ff = FeedForward(cfg)

        # normalization
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        # masking
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # 1

        # assgin input as shortcut
        shortcut = x

        # normalize input
        x = self.norm1(x)

        # get context vector
        x = self.att(x)

        # dropout
        x = self.drop_shortcut(x)

        # shortcut: add input to output 
        x = x + shortcut # 2

        # assgin transformed input to shortcut 
        shortcut = x # 3

        # normalizing
        x = self.norm2(x)

        # apply linear layers and activation functions to input
        x = self.ff(x)

        # drop
        x = self.drop_shortcut(x)

        # shortcut: add input to output 
        x = x + shortcut # 4

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # create token embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], cfg["emb_dim"])

        # create positional embeddings
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # set drop out rate
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Apply transfomer block with n_layers
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Apply layer normalization to embedding layers
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # create output layers
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias = False
        )
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # create token embeddings
        tok_embeds = self.tok_emb(in_idx)

        # create positional embeddings
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device = in_idx.device) # 1
        )
        
        # combine token and positional embeddings
        x = tok_embeds + pos_embeds

        # drop some layers
        x = self.drop_emb(x)

        # apply transformer blocksbb
        x = self.trf_blocks(x)

        # normalizing
        x = self.final_norm(x)

        # apply linear function to x and return probbaility of each token and text
        logits = self.out_head(x)
        
        return logits


import numpy as np

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                         "Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):           
    #1 Sets the model’s positional and token embedding weights to those specified in params.
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    #2 Iterates over each transformer block in the model
    for b in range(len(params["blocks"])):

        #3 The np.split function is used to divide the attention and bias weights into 
        # three equal parts for the query, key, and value components.

        # attention weights
        q_w, k_w, v_w = np.split(                            
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # bias weights
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # weight tensor for the output projection layer
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # weight and bias from layers
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # normalizing 
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    #4 he original GPT-2 model by OpenAI reused the token embedding weights in the output layer to 
    # reduce the total number of parameters, which is a concept known as weight tying.
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def generate(model, idx, max_new_tokens, context_size,
             temperatures = 0.0, top_k = None, eos_id = None):
    # 1 The for loop is the same as before: gets logits and only focuses on the last time step.
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        if top_k is not None:
           # 2 Filters logits with top_k sampling
           top_logits, _ = torch.topk(logits, top_k)
           min_val = top_logits[:, -1]
           logits = torch.where(
               logits < min_val,
               torch.tensor(float("-inf")).to(logits.device),
               logits
           )
        
        if temperatures > 0.0:
            # 3 Applies temperature scaling
            logits = logits / temperatures
            probas = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probas, num_samples = 1)
        else:
            # 4 Carries out greedy next-token selection as before when temperature scaling is disabled
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        
        # 5 Stops generating early if end-of-sequence token is encountered
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim = 1)
    
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    # 1
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # 2
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)         #1
    target_batch = target_batch.to(device)      
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

#1 Iteratives over all batches if no fixed num_batches is specified
#2 Reduces the number of batches to match the total number of batches in the data loader if num_batches exceeds the number of batches in the data loader
#3 Sums loss for each batch
#4 Averages the loss over all batches

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)     #1
    else:
        num_batches = min(num_batches, len(data_loader))   #2
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()    #3
        else:
            break
    return total_loss / num_batches    #4


def evaluate_model(model, train_loader, val_loader, 
                   device, eval_iter):
    # 1 Dropout is disabled during evaluation for stable, reproducible results.
    model.eval()
    with torch.no_grad():
        # 2 Disables gradient tracking, which is not required during evaluation, 
        #   to reduce the computational overhead
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches = eval_iter
        )
        
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches = eval_iter
        )
    
    model.train()
    return train_loss, val_loss


def generate_text_simple(model, idx,  # 1
                         max_new_tokens, context_size):
    
    # iterate number of max new tokens provided
    for _ in range(max_new_tokens):

        # extract last number of context size
        idx_cond = idx[:, -context_size:] # 2

        # Disables gradient tracking since we are not training yet
        with torch.no_grad():
            # Obtain logits
            logits = model(idx_cond)

        # only extract the last row from a tensor
        logits = logits[:, -1, :] # 3

        # Obtain probability through softmax
        # Probability of each token in vocabulary
        probas = torch.softmax(logits, dim = -1) # 4
        
        # find the max probability
        idx_next = torch.argmax(probas, dim = -1, keepdim = True) # 5
        
        # find the index corresponding to the max proba
        idx = torch.cat((idx, idx_next), dim = 1) # 6

    return idx

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(model = model, idx = encoded, 
                                         max_new_tokens = 50, context_size = context_size)
        
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    
    # 1 Compact print format
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, val_loader, 
                       optimizer, device, num_epochs, 
                       eval_freq, eval_iter, start_context, 
                       tokenizer):
    # 1 Initializes lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # 2 Starts the main training loop
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            # 3 Resets loss gradients from the previous batch iteration
            optimizer.zero_grad()

            # calculate loss value over each batch
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            # 4 Calculates loss gradients
            loss.backward()

            # 5 Updates model weights using loss gradients
            optimizer.step()

            # numel(): returns the total number of elements in the input tensor
            tokens_seen += input_batch.numel()

            global_step += 1

            # 6 Optional evaluation step
            if global_step % eval_freq == 0:

                # evaluate model by trainning loss and validation loss
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                print(f"Ep {epoch + 1} (Step {global_step: 06d}): "
                      f"Train loss {train_loss: .3f}",
                      f"Val loss {val_loss: .3f}")
                
        # 7 Prints a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    
    return train_losses, val_losses, track_tokens_seen