from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm import xLSTMLMModel, xLSTMLMModelConfig
import torch
from torch import nn

class xLSTMStockModel(xLSTMLMModel):
    def __init__(self,config: xLSTMLMModelConfig,**kwargs):
        super().__init__(config)
        self.token_embedding = nn.Linear(config.vocab_size,config.embedding_dim)


def xLSTMStockModeling(
    vocab_size=5,
    embedding_dim=128,
    context_lenght=256,
    act_fn= "gelu",
):
    xlstm_cfg = f""" 
vocab_size: {vocab_size}
mlstm_block:
  mlstm:
    conv1d_kernel_size: 4
    qkv_proj_blocksize: 4
    num_heads: 4
slstm_block:
  slstm:
    backend: cuda
    num_heads: 4
    conv1d_kernel_size: 4
    bias_init: powerlaw_blockdependent
  feedforward:
    proj_factor: 1.3
    act_fn: {act_fn}
context_length: {context_lenght}
num_blocks: 7
embedding_dim: {embedding_dim}
slstm_at: [1]
"""
    cfg = OmegaConf.create(xlstm_cfg)
    cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
    xlstm_stack = xLSTMStockModel(cfg)
    return xlstm_stack


if __name__ == '__main__':
    batch=10
    seqlen= 128
    ohlcv=5

    x = torch.rand(batch,seqlen,ohlcv).to("cuda")
    xlstm_stack = xLSTMStockModeling().to("cuda")
    y = xlstm_stack(x)
    print(y.shape)

