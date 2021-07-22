""" Module containing classes that interfaces neural network policies
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.functional.F as F
import pandas as pd

import pickle
from test.chem import RetroReaction_RL as RetroReaction
from test.utils.models import load_model
from test.models.utils import load_torch_model
from test.context.collection import ContextCollection
from test.utils.logging import logger, setup_logger
from test.utils.time_checker import DebugTimeCheck

from test.a2c.test2 import Node

if TYPE_CHECKING:
    from test.utils.type_utils import Union, Any, Sequence, List, Tuple
    from test.context.config import Configuration
    from test.chem import TreeMolecule

import logging

class PolicyException(Exception):
    """An exception raised by the Policy classes"""


def _make_fingerprint(
    obj: Union[TreeMolecule, RetroReaction], model: Any
    ) -> torch.Tensor:
    fingerprint = obj.fingerprint(radius=2, nbits=len(model))
    return fingerprint.reshape([1, len(model)])

class PairPolicy(ContextCollection):
    """
    An abstraction of a filter policy.

    This policy provides a query on a reaction to determine whether it is feasible

    :param config: the configuration of the tree search
    """

    _single_selection = True
    _collection_name = "pair policy"

    def __init__(self, config: Configuration) -> None:
        super().__init__()
        self._config = config

    def __call__(self, reaction: RetroReaction) -> bool:
        return self._predict(reaction)

    def _predict(self, reaction: RetroReaction) -> float:
        if not isinstance(self.selection, str):
            raise PolicyException("No policy selected.")

        model = self[self.selection]["model"]
        prod_fp, rxn_fp = self._reaction_to_fingerprint(reaction, model)
        return model(prod_fp, rxn_fp)

    @staticmethod
    def _reaction_to_fingerprint(
        reaction: RetroReaction, model: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        rxn_fp = _make_fingerprint(reaction, model)
        prod_fp = _make_fingerprint(reaction.mol, model)
        prod_fp = torch.LongTensor(prod_fp).repeat(len(rxn_fp),1)
        return prod_fp, rxn_fp

    def load(self, model_config_file, key: str) -> None:  # type: ignore
        """
        Load a policy under the given key

        If `source` is a string, it is taken as a path to a filename and the
        policy is loaded as an `LocalKerasModel` object.

        If `source` is not a string, it is taken as a custom object that
        implements the `__len__` and `predict` methods.

        :param source: the source of the policy model
        :param key: the key or label
        """
        self._logger.info(f"Loading filter policy model from {model_config_file} to {key}")
        model = load_torch_model(model_config_file)
        self._items[key] = {'model' : model}

    def load_from_config(self, **config: Any) -> None:
        """
        Load one or more filter policy from a configuration

        The format should be:
            key: path_to_model

        :param config: the configuration
        """
        for key, filename in config.items():
            self.load(filename, key)


class TemplatePolicy(ContextCollection):
    """
    An abstraction of an expansion policy.

    This policy provides actions (templates) that can be applied to a molecule

    :param config: the configuration of the tree search
    """
    logger = logging.getLogger()
    _collection_name = "template policy"
    
    def __init__(self, config: Configuration) -> None:
        super().__init__()
        self._config = config
        self._stock = config.stock

    def __call__(
        self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[Sequence[RetroReaction], Sequence[float]]:
        return self.get_actions(molecules)
    
    def get_actions(
            self, molecules: Sequence[TreeMolecule], context_mask: torch.Tensor
    ) -> Tuple[dict[RetroReaction], torch.Tensor]:
        """
        Get all the probable actions of a set of molecules, using the selected policies and given cutoffs

        :param molecules: the molecules to consider
        :return: the actions and the priors of those actions
        :raises: PolicyException: if the policy isn't selected
        """
        if not self.selection:
            raise PolicyException("No expansion policy selected")

        possible_actions = {}
        priors = []
        
        policy_key = self.selection[0]
        model = self[policy_key]["model"]
        templates = self[policy_key]["templates"]

        all_transforms_prob = self._predict_node(molecules, model)
        
        cutoff_mask = self._cutoff_predictions(F.softmax(all_transforms_prob,dim=-1))
        action_mask = cutoff_mask & context_mask
        _mol,_template = torch.where(action_mask)
        for idx_mol,idx_template in zip(_mol,_template):
            idx_mol = idx_mol.item(); idx_template = idx_template.item();
            metadata = dict(templates.iloc[idx_template])
            template = metadata[self._config.template_column]
            del metadata[self._config.template_column]
            action = RetroReaction(molecules[idx_mol], 
                                   template,
                                   metadata=metadata)
            possible_actions[(idx_mol,idx_template)] = action

        return possible_actions, all_transforms_prob, cutoff_mask

    def load(self, model_config_file, templatefile: str, key: str) -> None:  # type: ignore
        """
        Load a policy and associated templates under the given key

        If `source` is a string, it is taken as a path to a filename and the
        policy is loaded as an `LocalKerasModel` object.

        If `source` is not a string, it is taken as a custom object that
        implements the `__len__` and `predict` methods.

        :param source: the source of the policy model
        :param templatefile: the path to a HDF5 file with the templates
        :param key: the key or label
        :raises PolicyException: if the length of the model output vector is not same as the number of templates
        """
        self._logger.info(f"Loading expansion policy model from {model_config_file} to {key}")
        #if 'torch' in key:
        model = load_torch_model(model_config_file)
        self._logger.info(f"Loading templates from {templatefile} to {key}")

        with open(templatefile,'rb') as f:
            templates = pickle.load(f)

        if hasattr(model, "output_size") and len(templates) != model.output_size:  # type: ignore
            raise PolicyException(
                f"The number of templates ({len(templates)}) does not agree with the "  # type: ignore
                f"output dimensions of the model ({model.output_size})"
            )

        self._items[key] = {"model": model, "templates": templates}

    def load_from_config(self, **config: Any) -> None:
        """
        Load one or more expansion policy from a configuration

        The format should be
        key:
            - path_to_model
            - path_to_templates

        :param config: the configuration
        """
        for key, policy_spec in config.items():
            modelfile, templatefile = policy_spec
            self.load(modelfile, templatefile, key)

    def _cutoff_predictions(self, raw_prob: torch.Tensor) -> np.ndarray:
        """
        Get the top transformations, by selecting those that have:
            * cumulative probability less than a threshold (cutoff_cumulative)
            * or at most N (cutoff_number)
        """
        cutoff_mask = torch.zeros(raw_prob.size()).bool()
        prob, idx = torch.topk(F.softmax(raw_prob,dim =-1), 
                                self._config.cutoff_number,)
        cum_prob = torch.cumsum(prob, dim=-1,)
        def _last_false_to_true(d1_tensor):
            for i, value in enumerate(d1_tensor):
                if not value:
                    d1_tensor[i] = True
                    break

        for i,cum_prob_,template_idx_ in enumerate(zip(cum_prob,idx)):
            cum_prob_ = cum_prob_ < self._config.cutoff_cumulative
            _last_false_to_true(cum_prob_)
            template_idx_ = template_idx_[cum_prob_]
            for j in template_idx_:
                cutoff_mask[i,j] = True
        
        return cutoff_mask


    @staticmethod
    def _predict(mol: TreeMolecule, model: Any) -> torch.Tensor:
        fp_arr = _make_fingerprint(mol, model)
        return model(torch.Tensor(fp_arr)).flatten()

    @staticmethod
    def _predict_node(molecules: Sequence[TreeMolecule], model: Any) \
            -> torch.Tensor:
        """
        node with n_mols -> output shape (n_mols x n_templates)
        """
        fp_arr = []
        for mol in molecules:
            fp_arr.append(_make_fingerprint(mol, model))
        fp_arr = torch.Tensor(fp_arr)
        return model(fp_arr)

