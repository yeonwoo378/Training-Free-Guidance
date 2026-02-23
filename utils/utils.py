import torch
import os
import numpy as np
from typing import Union
from transformers import HfArgumentParser

from .configs import Arguments

from evaluations.image import ImageEvaluator
from evaluations.molecule import MoleculeEvaluator
from evaluations.audio import AudioEvaluator

from diffusion.ddim import ImageSampler, MoleculeSampler
from diffusion.audio_diffusion import AudioDiffusionSampler
from diffusion.stable_diffusion import StableDiffusionSampler

from methods.mpgd import MPGDGuidance
from methods.lgd import LGDGuidance
from methods.base import BaseGuidance
from methods.ugd import UGDGuidance
from methods.freedom import FreedomGuidance
from methods.dps import DPSGuidance
from methods.tfg import TFGGuidance
from methods.cg import ClassifierGuidance


import pickle


def get_logging_dir(arg_dict: dict):
    if arg_dict['guidance_name'] == 'tfg':
        # record rho, mu, sigma with scheduler
        suffix = f"rho={arg_dict['rho']}-{arg_dict['rho_schedule']}+mu={arg_dict['mu']}-{arg_dict['mu_schedule']}+sigma={arg_dict['sigma']}-{arg_dict['sigma_schedule']}"
    else:
        suffix = "guidance_strength=" + str(arg_dict['guidance_strength'])
    
    return os.path.join(
        arg_dict['logging_dir'],
        f"guidance_name={arg_dict['guidance_name']}+recur_steps={arg_dict['recur_steps']}+iter_steps={arg_dict['iter_steps']}",
        "model=" + arg_dict['model_name_or_path'].replace("/", '_'),
        "guide_net=" + arg_dict['guide_network'].replace('/', '_'),
        "target=" + str(arg_dict['target']).replace(" ", "_"),
        "bon=" + str(arg_dict['bon_rate']),
        suffix,
    )

def get_config(add_logger=True) -> Arguments:
    args = HfArgumentParser([Arguments]).parse_args_into_dataclasses()[0]
    args.device = torch.device(args.device)

    if add_logger:
        from logger import setup_logger
        args.logging_dir = get_logging_dir(vars(args))
        print("logging to", args.logging_dir)
        setup_logger(args)
    
    if args.data_type == 'molecule':
        # load args
        def _get_args_gen(args_path, argse_path):
            with open(args_path, 'rb') as f:
                args_gen = pickle.load(f)
            assert args_gen.dataset == 'qm9_second_half'

            with open(argse_path, 'rb') as f:
                args_en = pickle.load(f)

            # Add missing args!
            if not hasattr(args_gen, 'normalization_factor'):
                args_gen.normalization_factor = 1
            if not hasattr(args_gen, 'aggregation_method'):
                args_gen.aggregation_method = 'sum'

            return args_gen, args_en

        args.args_gen, args.args_en = _get_args_gen(args.args_generators_path, args.args_energy_path)
    
    # examine combined guidance

    args.tasks = args.task.split('+')
    args.guide_networks = args.guide_network.split('+')
    args.targets = args.target.split('+')

    assert len(args.tasks) == len(args.guide_networks) == len(args.targets)

    return args


def get_evaluator(args):

    if args.data_type == 'image':
        return ImageEvaluator(args)
    elif args.data_type == 'molecule':
        return MoleculeEvaluator(args)
    elif args.data_type == 'text2image':
        return ImageEvaluator(args)
    elif args.data_type == 'audio':
        return AudioEvaluator(args)
    else:
        raise NotImplementedError

def get_guidance(args, network):
    noise_fn = getattr(network, 'noise_fn', None)
    if args.guidance_name == 'no':
        return BaseGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'mpgd':
        return MPGDGuidance(args, noise_fn=noise_fn)
    elif 'ugd' in args.guidance_name:
        return UGDGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'freedom':
        return FreedomGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'dps':
        return DPSGuidance(args, noise_fn=noise_fn)
    elif 'lgd' in args.guidance_name:
        return LGDGuidance(args, noise_fn=noise_fn)
    elif "tfg" in args.guidance_name:
        return TFGGuidance(args, noise_fn=noise_fn)
    elif 'cg' in args.guidance_name:
        return ClassifierGuidance(args, noise_fn=noise_fn)
    else:
        raise NotImplementedError

def get_network(args):
    
    if args.data_type == 'image':
        return ImageSampler(args)
    elif args.data_type == 'molecule':
        return MoleculeSampler(args)
    elif args.data_type == 'text2image':
        return StableDiffusionSampler(args)
    elif args.data_type == 'audio':
        return AudioDiffusionSampler(args)
    else:
        raise NotImplementedError


# @torch.no_grad()
def divergence_stepper( v_func,
                        v_func_kwargs,
                        x_key='z',
                        t_key='t',
                        stop_t=0.5,
                        num_updates=1,
                        num_delta=1,
                        num_eps=1,
                        delta_scale=0.1,
                        delta_scheduler=lambda t: 2 ** (-t),
                        seed_delta=None,
                        seed_eps=None,
                        delta=None,
                        improved=None,
                        resample_delta=False,
                        resample_eps=False,
                        sequential_vjp=True,
                        sequential_hutchinson=True,
                        eta=0.0,
                        sync_over_time=True
                        ):
    assert stop_t >= 0.0 and stop_t <= 1.0

    t = v_func_kwargs[t_key]
    # import ipdb; ipdb.set_trace()
    # if isinstance(t, torch.Tensor):
    # t is long 
        # assert (t == t.mean()).all().item(), "All timesteps in the batch must be the same for divergence_stepper."
    t = 1. - (t+1)/1000
    # import ipdb; ipdb.set_trace()
    if num_updates <= 0 or t > stop_t:
        return v_func_kwargs[x_key], v_func(**v_func_kwargs), improved, delta
    # import ipdb; ipdb.set_trace()
    z = v_func_kwargs[x_key]        
    B = z.shape[0]
    D = np.prod(z.shape[1:])  # C * H * W
    
    delta_generator = None
    eps_generator = None
    
    if seed_delta is not None:
        if sync_over_time:
            delta_generator = torch.Generator(device=z.device).manual_seed(seed_delta) # + int(t * 1000))
        else:
            delta_generator = torch.Generator(device=z.device).manual_seed(seed_delta + int(t * 1000))
    if seed_eps is not None:
        eps_generator = torch.Generator(device=z.device).manual_seed(seed_eps)
    sync_eps_with_delta = num_eps == 1 and seed_eps == seed_delta
    
    for update_idx in range(num_updates):
        require_sample_delta = (update_idx == 0) or resample_delta
        require_sample_eps = (update_idx == 0) or resample_eps

        # compute divergence and find the best perturbation
        if sequential_vjp:
            assert (not resample_delta) or (num_delta==1)
            for delta_idx in range(num_delta+1):

                # pass if no need to get the divergence of original z
                # if delta_idx == 0 and update_idx !=0: # buggy
                #     continue
                # import ipdb; ipdb.set_trace()
                if delta is None or improved is None:
                    assert improved is None and delta_idx <= 1
                    delta = torch.randn(z.shape, generator=delta_generator, device=z.device) # if delta_idx != 0 else torch.zeros_like(z, device=z.device)
                elif update_idx > 0:
                    temp_delta_generator = torch.Generator(device=z.device).manual_seed(seed_delta + update_idx)
                    temp_delta = torch.randn(z.shape, generator=temp_delta_generator, device=z.device)
                    pass
                elif delta_idx > 0:
                    new_delta = torch.randn(z.shape, generator=delta_generator, device=z.device) #if delta_idx != 0 else torch.zeros_like(z, device=z.device)
                    delta = torch.where(
                        improved.reshape(-1, *([1]*(z.ndim-1))), # hard-coded shape
                        delta, # True
                        new_delta # False
                    )
                # no update delta when delta_idx=0
                assert seed_delta != seed_eps, "Is a Biased Estimator"

                if sync_eps_with_delta and delta_idx != 0:
                    eps = delta.detach()
                    raise NotImplementedError # not using anymore!

                else:
                    eps = torch.randn(z.shape, generator=eps_generator, device=z.device) 

                if delta_idx == 0:
                    perturbed_z = z 
                elif update_idx == 0:
                    perturbed_z = z + delta_scale * (1. - t) * delta_scheduler(update_idx) * delta # TODO: clarify
                else:
                    perturbed_z = z + delta_scale * (1. - t) * delta_scheduler(update_idx) * temp_delta # TODO: clarify
                with torch.enable_grad():
                    perturbed_z = perturbed_z.detach().requires_grad_(True)
                    v_func_kwargs[x_key] = perturbed_z
                    
                    v_pred = v_func(**v_func_kwargs)  # [B, C, H, W]
                    v_pred_eps = (v_pred * eps).flatten(1).sum(1)  # [B]
                    grad_v = torch.autograd.grad(
                        outputs=v_pred_eps,          # [B]
                        inputs=perturbed_z,                      # [B, C, H, W]
                        grad_outputs=torch.ones_like(v_pred_eps),  # [B]
                        create_graph=False,
                        retain_graph=False,         
                    )[0].detach()  # [B, C, H, W]
                    divergence = (grad_v * eps).flatten(1).sum(1) / D  # [B]
                
                threshold = - (1 / (1 - t))

                if delta_idx == 0:
                    best_divergence = divergence.detach()
                    best_v_pred = v_pred.detach()
                    best_perturbed_z = perturbed_z.detach()
                elif update_idx == 0:
                    improved = (divergence < (best_divergence - eta)) & (best_divergence >= threshold)
                    improved_shape = (B,) + (1,) * (len(z.shape) - 1)
                    best_divergence = torch.where(improved, divergence, best_divergence)
                    best_v_pred = torch.where(
                        improved.view(improved_shape),
                        v_pred,
                        best_v_pred,
                    )
                    # print(improved.view(improved_shape).shape)
                    best_perturbed_z = torch.where(
                        improved.view(improved_shape),
                        perturbed_z.detach(),
                        best_perturbed_z,
                    )
                else:
                    temp_improved = (divergence < (best_divergence - eta)) & (best_divergence >= threshold)
                    improved_shape = (B,) + (1,) * (len(z.shape) - 1)
                    best_divergence = torch.where(temp_improved, divergence, best_divergence)
                    best_v_pred = torch.where(
                        temp_improved.view(improved_shape),
                        v_pred,
                        best_v_pred,
                    )
                    # print(improved.view(improved_shape).shape)
                    best_perturbed_z = torch.where(
                        temp_improved.view(improved_shape),
                        perturbed_z.detach(),
                        best_perturbed_z,
                    )

                    # improved = improved & temp_improved

            # update iteration-wise
            z = best_perturbed_z # update z
            v_pred = best_v_pred
        
        # currently not using hereafter
        else:
            # build delta
            raise NotImplementedError
           
    return best_perturbed_z, best_v_pred, improved, delta

