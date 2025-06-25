"""
---
title: Sampling Techniques for Language Models
summary: >
 A set of PyTorch implementations/tutorials of sampling techniques for language models.
---

# Sampling Techniques for Language Models

* [Greedy Sampling]
* [Temperature Sampling]
* [Top-k Sampling]
* [Nucleus Sampling]
* [Random Sampling]


"""

import torch


class Sampler:
    """
    ### Sampler base class
    """

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        ### Sample from logits

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        """
        raise NotImplementedError()
