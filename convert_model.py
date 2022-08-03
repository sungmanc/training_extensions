import torch

ckpt_path = '/home/sungmanc/scripts/pretrained_weights/atss_metanet_12ep_newnorm.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')

new_ckpt = {'meta':ckpt['meta'], 'model':ckpt['state_dict']}
torch.save(new_ckpt, '/home/sungmanc/scripts/pretrained_weights/atss_metanet_12ep_newnorm_ote.pth')