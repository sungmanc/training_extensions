import torch

ckpt_path = '/home/sungmanc/scripts/pretrained_weights/in1k_gcc15m.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')

new_ckpt = {'model':{}}
for k, v in ckpt['model'].items():
    if k.startswith('image_encoder'):
        new_key = k.replace('image_encoder', 'backbone')
        new_ckpt['model'][new_key] = v
        print('{} --> {}'.format(k, new_key))

torch.save(new_ckpt, '/home/sungmanc/scripts/pretrained_weights/unicl_swin_tiny_in1k_gcc15m.pth')


'''
ckpt_path = '/home/sungmanc/scripts/pretrained_weights/swin_tiny_patch4_window7_224_in1k.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')

new_ckpt = {'model':{}}
for k, v in ckpt['model'].items():
    new_key = 'backbone.' + k
    new_ckpt['model'][new_key] = v
    print('{} --> {}'.format(k, new_key))

torch.save(new_ckpt, '/home/sungmanc/scripts/pretrained_weights/supervised_swin_tiny_1k.pth')
'''