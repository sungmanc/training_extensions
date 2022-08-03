import torch


ckpt_path = '/home/sungmanc/scripts/ote/training_extensions/outputs/large_upm_det_benchmark/few_shot_lp/metanet_b4_48ep_bccd_0.1_1/results/env_model_ckpt.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')

ref_ckpt_path = '/home/sungmanc/scripts/mmdetection/work_dirs/atss_metanet_b4_fpn_4x_coco/epoch_48.pth'
#ref_ckpt_path = '/home/sungmanc/scripts/mmdetection/logs/metanet_b4_1x_coco/epoch_12.pth'
ref_ckpt = torch.load(ref_ckpt_path, map_location='cpu')['state_dict']



ref_key_list = list(ref_ckpt.keys())
for i, (k, v) in enumerate(ckpt.items()):
    src_key = k
    ref_key = ref_key_list[i]

    print('{}, {}'.format(src_key, torch.all(torch.eq(v, ref_ckpt[ref_key]))))
