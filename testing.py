#%%
from yahpo_train.cont_normalization import ContNormalization
from yahpo_train.model  import *
from yahpo_gym import cfg
from yahpo_train.metrics import *
from yahpo_gym.benchmarks import lcbench, rbv2
from yahpo_gym.local_config import local_config


# %%
local_config.init_config("C:\\Users\\svenm\\LRZ Sync+Share\\multifidelity_data (Florian Pfisterer)")
cc = cfg('rbv2_glmnet')


# %%
dls = dl_from_config(cc, bs=2048)
# %%
