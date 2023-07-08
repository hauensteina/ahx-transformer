import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import tensor

"""
Transformer as suggested in "Let's build GPT from scratch" by Andrej Karpathy.
https://www.youtube.com/watch?v=kCc8FmEb1nY
B: Batch element dimension
T: Time dimension, range(BLOCK_SZ), which is the context length
C: Channel dimension, which is the embedding length or in general, the number of output logits of a layer
"""

class TransformerModel(nn.Module):
    def __init__(self, device, tokenizer, embed_sz, num_layers, num_heads, block_sz, dropout):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.embed_sz = embed_sz
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.block_sz = block_sz
        self.dropout = dropout
        # For each char, store the probs of the next char
        self.token_embedding_table = nn.Embedding(self.tokenizer.vocab_sz, self.embed_sz)
        self.position_embedding_table = nn.Embedding(block_sz, self.embed_sz)
        head_sz = self.embed_sz//self.num_heads
        self.blocks = nn.Sequential(
            *[Block(num_heads, embed_sz , block_sz, head_sz, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(self.embed_sz,elementwise_affine=True)
        self.lm_head = nn.Linear(self.embed_sz, self.tokenizer.vocab_sz)

    def add_optimizer(self, lr):
        self.learning_rate = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, inp, targets=None):
        """ nn.Module.__call__() calls forward(). """
        B,T = inp.shape
        tok_emb = self.token_embedding_table( inp) # (B,T,C)
        pos_emb = self.position_embedding_table( torch.arange( T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
            return logits, loss
        else:
            # Pytorch wants the logits to be (B,C,T) for cross_entropy
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Targets is an index that is supposed to be largest in the logits vector,
            # for each token in each input of the batch.
            loss = F.cross_entropy(logits, targets)
            return logits, loss
    
    @torch.no_grad()
    def generate(self, prompt, stoptoken=None, max_new_tokens=100):
        """ Generate from a prompt """

        lprompt = self.tokenizer.encode(prompt)
        # Add a fake batch dimension
        lprompt = torch.tensor(lprompt, dtype=torch.long).unsqueeze(0)
        lprompt = lprompt.to(self.device)

        for _ in range(max_new_tokens):
            # get the predictions
            lprompt = lprompt[:, -self.block_sz:] # (B,block_sz) limit input to block_sz
            logits, loss = self(lprompt)
            logits = logits[:,-1,:] # B,C because we only take the last token
            probs = F.softmax(logits, dim=-1)
            #next = torch.multinomial(probs, num_samples=1) # pull from distribution
            _,next = torch.max(probs, dim=1) # Just take the most likely
            next = next.unsqueeze(-1)
            lprompt = torch.cat([lprompt, next], dim=1)
            if next[0].tolist() == stoptoken:
                break
        out = self.tokenizer.decode(lprompt[0].tolist())
        return out
    
    def save(self, fname, infodict={}):
        """ Save the model plus optimizer state so we can resume training later """
        hyper_parameters = {
            # How many stacked transformer blocks.
            'num_layers': self.num_layers,
            # Each block takes and produces vectors of this size.
            'embed_sz': self.embed_sz,
            # Number of parallel heads in each layer.
            # Concat head outputs to get back to embed_sz.
            'num_heads': self.num_heads,
            # T = block_sz (aka context length) during training .
            # During inference, block_sz is just an upper limit on T.
            'block_sz': self.block_sz,
            'dropout': self.dropout,
            }
        optimizer_parameters = { 
            'lr': self.learning_rate,
            }
        torch.save({
            'infodict': infodict,
            'hyper_parameters': hyper_parameters,
            'optimizer_parameters': optimizer_parameters,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer_dict': self.tokenizer.ddict(),
            }, fname)
        
    @classmethod 
    def load(cls, device, tokenizer, fname):
        """ Load the model,optimizer,tokenizer from a checkpoint file """
        checkpoint = torch.load(fname)
        tokenizer.load_dict(checkpoint['tokenizer_dict'])
        model = cls( device, tokenizer, **checkpoint['hyper_parameters'])
        model.load_state_dict(checkpoint['model_state_dict'])
        m = model.to(device)
        m.add_optimizer(**checkpoint['optimizer_parameters'])    
        m.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return m

class Head(nn.Module):
    """ One head of self attention """
    def __init__(self,embed_sz,block_sz,head_sz, dropout):
        super().__init__()
        self.key = nn.Linear(embed_sz, head_sz, bias=False)
        self.query = nn.Linear(embed_sz, head_sz, bias=False)
        self.value = nn.Linear(embed_sz, head_sz, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_sz,block_sz))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_sz)
        q = self.query(x) # (B,T,head_sz)
        v = self.value(x) # (B,T,head_sz)
        # compute affinities
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,T)
        # Decoders only look into the past, so we mask the future
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T,T)

        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei) # maybe apply this to wei @ v instead?

        out = wei @ v # (B,T,T) @ (B,T,head_sz) --> (B,T,head_sz)
        return out

class MultiHead(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, embed_sz, block_sz, head_sz, dropout):
        super().__init__()
        self.heads = nn.ModuleList( [Head(embed_sz, block_sz, head_sz, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear( embed_sz, embed_sz)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a nonlinearity """
    def __init__(self, n_inout, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inout, 4 * n_inout),
            nn.ReLU(),
            nn.Linear( 4 * n_inout, n_inout),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication (sa_heads) followed by computation (ffwd) """
    def __init__(self, num_heads, embed_sz, block_sz, head_sz, dropout):
        super().__init__()
        self.sa_heads = MultiHead(num_heads, embed_sz, block_sz, head_sz, dropout)
        self.ffwd = FeedForward(embed_sz, dropout)    
        self.ln1 = nn.LayerNorm(embed_sz,elementwise_affine=True)
        self.ln2 = nn.LayerNorm(embed_sz,elementwise_affine=True)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x    

