import torch
import os
import numpy as np
import PIL.Image as Image
from abc import ABC, abstractmethod
from diffusion.base import BaseSampler
from methods.base import BaseGuidance
from evaluations.base import BaseEvaluator
from utils.configs import Arguments
import logger

class BasePipeline(object):
    def __init__(self,
                 args: Arguments, 
                 network: BaseSampler, 
                 guider: BaseGuidance, 
                 evaluator: BaseEvaluator,
                 bon_guider=None):
        self.network = network
        self.guider = guider
        self.evaluator = evaluator
        self.logging_dir = args.logging_dir
        self.check_done = args.check_done
        
        self.bon_rate = args.bon_rate
        self.batch_size = args.eval_batch_size
        
        # 初始化 logp_guider，如果没有提供则使用默认的 guider
        self.bon_guider = bon_guider if bon_guider is not None else self.guider
        
    @abstractmethod
    def sample(self, sample_size: int):
        
        # this makes resue the previous results, often ignore the sample size
        # samples = self.check_done_and_load_sample()
        samples = None
        
        if samples is None:

            guidance_batch_size = self.batch_size  

            samples = self.network.sample(sample_size=sample_size * self.bon_rate, guidance=self.guider)

            logp_list = []
            for i in range(0, samples.shape[0], guidance_batch_size):
                batch_samples = samples[i:i + guidance_batch_size]
                batch_logp = self.bon_guider.guider.get_guidance(batch_samples, return_logp=True, check_grad=False)
                logp_list.append(batch_logp)

            logp = torch.cat(logp_list, dim=0).view(-1)

            samples = samples.view(sample_size, int(self.bon_rate), *samples.shape[1:])
            logp = logp.view(sample_size, int(self.bon_rate))

            idx = logp.argmax(dim=1)
            samples = samples[torch.arange(sample_size), idx]

            samples = self.network.tensor_to_obj(samples)
                    
        return samples
    
    def evaluate(self, samples):
        return self.check_done_and_evaluate(samples)
    
    def check_done_and_evaluate(self, samples):
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, 'metrics.json')):
            logger.log("Metrics already generated. To regenerate, please set `check_done` to `False`.")
            return None
        return self.evaluator.evaluate(samples)

    def check_done_and_load_sample(self):
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, "finished_sampling")):
            logger.log("found tags for generated samples, should load directly. To regenerate, please set `check_done` to `False`.")
            return logger.load_samples()

        return None

