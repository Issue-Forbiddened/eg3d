
import legacy
import dnnlib
import torch

import argparse
import os


parser = argparse.ArgumentParser(description='Transform a .pkl file to a .pth file')
parser.add_argument('--pkl_path', type=str,
                    help='Path to the source .pkl file')
parser.add_argument('--pth_path', type=str, default='',
                    help='Path to the target .pth file')
parser.add_argument('--net_type', type=str, default='eg3d',
                    help='Type of the network to be transformed')
args = parser.parse_args()
assert args.net_type in ['eg3d','panohead']

if args.net_type=='eg3d':
    from training.triplane import TriPlaneGenerator
    from training.dual_discriminator import DualDiscriminator
    from torch_utils import misc

elif args.net_type=='panohead':
    import sys
    # sys.path.append('/root/PanoHead/training')
    # sys.path.append('/root/PanoHead/misc')
    # 在sys.path里最头部插入路径
    sys.path.insert(0, '/root/PanoHead/training')
    sys.path.insert(0, '/root/PanoHead')
    import triplane 
    TriPlaneGenerator=triplane.TriPlaneGenerator
    import dual_discriminator 
    DualDiscriminator=dual_discriminator.MaskDualDiscriminatorV2
    from torch_utils import misc


basemodel_path=args.pkl_path

tosave_path=args.pth_path if args.pth_path else basemodel_path.replace('.pkl','.pth')

print(f'Transforming {basemodel_path} to {tosave_path}, net_type: {args.net_type}')

device = torch.device('cuda:0')
with dnnlib.util.open_url(basemodel_path) as f:
    loaded_pickle=legacy.load_network_pkl(f)
    basemodel = loaded_pickle['G_ema'].to(device) # type: ignore
training_set_kwargs=loaded_pickle['training_set_kwargs']
basemodel_new = TriPlaneGenerator(*basemodel.init_args, **basemodel.init_kwargs).requires_grad_(False).to(device)
misc.copy_params_and_buffers(basemodel, basemodel_new, require_all=False)
basemodel_new.neural_rendering_resolution = basemodel.neural_rendering_resolution
basemodel_new.rendering_kwargs = basemodel.rendering_kwargs
basemodel = basemodel_new

D_init_args=loaded_pickle['D'].init_args
D_init_kwargs=loaded_pickle['D'].init_kwargs
D = DualDiscriminator(*D_init_args,**D_init_kwargs,).train().requires_grad_(False).to(device) # subclass of torch.nn.Module

tosave={}
# tosave['G_init_args']=eg3d_model['G']['init_args']
# tosave['G_init_kwargs']=eg3d_model['G']['init_kwargs']
# tosave['G_rendering_kwargs']=eg3d_model['G']['rendering_kwargs']
# tosave['G_neural_rendering_resolution']=eg3d_model['G']['neural_rendering_resolution']
# tosave['training_set_kwargs']=eg3d_model['training_set_kwargs']
tosave['G_init_args']=basemodel.init_args
tosave['G_init_kwargs']=basemodel.init_kwargs
tosave['G_rendering_kwargs']=basemodel.rendering_kwargs
tosave['G_neural_rendering_resolution']=basemodel.neural_rendering_resolution
tosave['training_set_kwargs']=training_set_kwargs
tosave['G']=basemodel.state_dict()

tosave['D_init_args']=D_init_args
tosave['D_init_kwargs']=D_init_kwargs
tosave['D']=D.state_dict()

torch.save(tosave, tosave_path)

loaded=torch.load(tosave_path)
pass
