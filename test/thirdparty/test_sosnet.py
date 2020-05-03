import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('sosnet') 

import torch
import sosnet_model
import os

tfeat_base_path='../../thirdparty/SOSNet/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

sosnet32 = sosnet_model.SOSNet32x32()
net_name = 'liberty'
sosnet32.load_state_dict(torch.load(os.path.join(tfeat_base_path, 'sosnet-weights', "sosnet-32x32-" + net_name + ".pth")))
sosnet32.cuda().eval()

patches = torch.rand(100, 1, 32, 32).to(device)
descrs = sosnet32(patches)

print('done!')