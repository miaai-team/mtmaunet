import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Transformer_Unit(nn.Module):
    def __init__(self, *, image_size, channels, dim=512, depth=4, heads=8, dim_head=64, mlp_dim=4096, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        length = image_height*image_width*image_depth
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv3d(channels,dim,3,1,1),
            Rearrange('b c h w d-> b (h w d) c'),
        )
        
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b (h w d) c-> b c h w d',h = image_height, w = image_width, d = image_depth),
            # nn.Conv3d(dim,channels,3,1,1),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, length, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)


    def forward(self, img):
        x = self.to_patch_embedding(img)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        return self.from_patch_embedding(x)
    
    

# net =Transformer_Unit(image_size=(16,16,16), channels=32, depth=2).cuda()

# x = torch.rand((1,32,16,16,16)).cuda()
# y = net(x)
# # print(net)
# print(y.shape)
