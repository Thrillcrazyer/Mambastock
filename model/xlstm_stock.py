from omegaconf import OmegaConf
from dacite import from_dict
import json
import os
from dacite import Config as DaciteConfig
from xlstm import xLSTMLMModel, xLSTMLMModelConfig
import torch
from torch import nn

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file


def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))


def load_state_dict_hf(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, map_location=mapped_device)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict
  
  
class xLSTMStockModel(xLSTMLMModel):
    def __init__(self,config: xLSTMLMModelConfig,**kwargs):
        super().__init__(config)
        self.token_embedding = nn.Linear(config.vocab_size,config.embedding_dim)
        self.config=config
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name,config=None, device=None, dtype=None, **kwargs):
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model
    
    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)


def xLSTMStockModeling(
    vocab_size=5,
    embedding_dim=768,
    context_lenght=128,
    num_block=7,
    act_fn= "gelu",
    pretrained_path=None
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
num_blocks: {num_block}
embedding_dim: {embedding_dim}
slstm_at: [1]
"""
    cfg = OmegaConf.create(xlstm_cfg)
    cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
    
    if pretrained_path is None:
        xlstm_stack = xLSTMStockModel(cfg)
    else:
        xlstm_stack = xLSTMStockModel(cfg).from_pretrained(pretrained_model_name=pretrained_path,config=cfg)
    return xlstm_stack


if __name__ == '__main__':
    batch=10
    seqlen= 128
    ohlcv=5

    x = torch.rand(batch,seqlen,ohlcv).to("cuda")
    model = xLSTMStockModeling().to("cuda")
    
    total_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Total model size in bytes: {total_memory} bytes")
    print(f"Total model size in megabytes: {total_memory / (1024 ** 2):.2f} MB")
    y = model(x)
    print(y.shape)

