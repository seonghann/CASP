""" Module containing classes that interfaces neural network policies
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

import pickle
from test.chem import RetroReaction
from test.utils.models import load_model
from test.models.utils import load_torch_model
from test.context.collection import ContextCollection
from test.utils.logging import logger, setup_logger
from test.utils.time_checker import DebugTimeCheck

from test.mcts.node import Node

if TYPE_CHECKING:
    from test.utils.type_utils import Union, Any, Sequence, List, Tuple
    from test.context.config import Configuration
    from test.chem import TreeMolecule

import logging

class PolicyException(Exception):
    """An exception raised by the Policy classes"""


def _make_fingerprint(
    obj: Union[TreeMolecule, RetroReaction], model: Any
) -> np.ndarray:
    fingerprint = obj.fingerprint(radius=2, nbits=len(model))
    return fingerprint.reshape([1, len(model)])

class ExpansionPolicy(ContextCollection):
    """
    An abstraction of an expansion policy.

    This policy provides actions (templates) that can be applied to a molecule

    :param config: the configuration of the tree search
    """
    logger = logging.getLogger()
    _collection_name = "expansion policy"
    
    def __init__(self, config: Configuration) -> None:
        super().__init__()
        self._config = config
        self._stock = config.stock

    def __call__(
        self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[Sequence[RetroReaction], Sequence[float]]:
        return self.get_actions(molecules)
    
    @DebugTimeCheck(logger, 'ExpansionPolicy.get_actions()')
    def get_actions(
        self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules, using the selected policies and given cutoffs

        :param molecules: the molecules to consider
        :return: the actions and the priors of those actions
        :raises: PolicyException: if the policy isn't selected
        """
        if not self.selection:
            raise PolicyException("No expansion policy selected")

        possible_actions = []
        priors = []

        for mol in molecules:
            if mol in self._stock:
                continue

            for policy_key in self.selection:
                model = self[policy_key]["model"]
                templates = self[policy_key]["templates"]
                all_transforms_prob = self._predict(mol, model)

                #all_transforms_prob = self._predict2(mol, model)
                probable_transforms_idx = self._cutoff_predictions(all_transforms_prob)
                possible_moves = templates.iloc[probable_transforms_idx]
                probs = all_transforms_prob[probable_transforms_idx]

                priors.extend(probs)
                for idx, (move_index, move) in enumerate(possible_moves.iterrows()):
                    metadata = dict(move)
                    del metadata[self._config.template_column]
                    metadata["policy_probability"] = float(probs[idx].round(4))
                    metadata["policy_name"] = policy_key
                    metadata["template_code"] = move_index
                    possible_actions.append(
                        RetroReaction(
                            mol, move[self._config.template_column], metadata=metadata
                        )
                    )

        return possible_actions, priors
    
    def get_actions2(
        
            self, molecules: Sequence[TreeMolecule], action_mask: torch.Tensor
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

        for idx_mol, (mol,mask) in enumerate(zip(molecules,action_mask)):
            all_transforms_prob = self._predict2(mol, model)
            # all_transforms_prob : Tensor , n_template 
            priors.append(all_transforms_prob)
            for idx_template, mask_value in enumerate(mask):
                if not mask_value:
                    continue
                else:
                    metadata = dict(templates.iloc[idx_template])
                    template = metadata[self._config.template_column]
                    del metadata[self._config.template_column]
                    action = RetroReaction(mol, 
                                  template,
                                  metadata=metadata)
                    possible_actions[(idx_mol,idx_template)] = action
        return possible_actions, torch.stack(priors)

    def load(self, source: Union[str, Any], templatefile: str, key: str) -> None:  # type: ignore
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
        self._logger.info(f"Loading expansion policy model from {source} to {key}")
        if 'torch' in key:
            model = load_torch_model(source,'expansion',device=self._config.device)
        else:
            model = load_model(source, key, self._config.use_remote_models)
        #else:
        #    model = load_model_from_local(source, key)

        self._logger.info(f"Loading templates from {templatefile} to {key}")
        if 'pkl' in templatefile or 'pck' in templatefile:
            with open(templatefile,'rb') as f:
                templates = pickle.load(f)
        else:
            templates: pd.DataFrame = pd.read_hdf(templatefile, "table")

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

    def _cutoff_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Get the top transformations, by selecting those that have:
            * cumulative probability less than a threshold (cutoff_cumulative)
            * or at most N (cutoff_number)
        """
        sortidx = np.argsort(predictions)[::-1]
        cumsum: np.ndarray = np.cumsum(predictions[sortidx])
        if any(cumsum >= self._config.cutoff_cumulative):
            maxidx = np.argmin(cumsum < self._config.cutoff_cumulative)
        else:
            maxidx = len(cumsum)
        maxidx = min(maxidx, self._config.cutoff_number) or 1
        return sortidx[:maxidx]

    @staticmethod
    def _predict(mol: TreeMolecule, model: Any) -> np.ndarray:
        fp_arr = _make_fingerprint(mol, model)
        if hasattr(model,'torchmodel'):
            return model(torch.Tensor(fp_arr)).flatten().detach().numpy()
        else:
            return np.array(model.predict(fp_arr)).flatten()
        
    @staticmethod
    def _predict2(mol: TreeMolecule, model: Any) -> np.ndarray:
        return model(None)

class FilterPolicy(ContextCollection):
    """
    An abstraction of a filter policy.

    This policy provides a query on a reaction to determine whether it is feasible

    :param config: the configuration of the tree search
    """

    _single_selection = True
    _collection_name = "filter policy"

    def __init__(self, config: Configuration) -> None:
        super().__init__()
        self._config = config

    def __call__(self, reaction: RetroReaction) -> bool:
        return self.is_feasible(reaction)

    def feasibility(self, reaction: RetroReaction) -> Tuple[bool, float]:
        """
        Computes if a given reaction is feasible by given
        the reaction fingerprint to a network model

        :param reaction: the reaction to query
        :return: if the reaction is feasible
        :raises: PolicyException: if the policy isn't selected
        """
        if not self._selection:
            raise PolicyException("No filter policy selected!")

        if not reaction.reactants:
            return False, 0.0

        prob = self._predict(reaction)
        feasible = prob >= self._config.filter_cutoff
        return feasible, prob

    def is_feasible(self, reaction: RetroReaction) -> bool:
        """
        Computes if a given reaction is feasible by given
        the reaction fingerprint to a network model

        :param reaction: the reaction to query
        :return: if the reaction is feasible
        :raises: PolicyException: if the policy isn't selected
        """
        feasible, _ = self.feasibility(reaction)
        return feasible

    def load(self, source: Union[str, Any], key: str) -> None:  # type: ignore
        """
        Load a policy under the given key

        If `source` is a string, it is taken as a path to a filename and the
        policy is loaded as an `LocalKerasModel` object.

        If `source` is not a string, it is taken as a custom object that
        implements the `__len__` and `predict` methods.

        :param source: the source of the policy model
        :param key: the key or label
        """
        self._logger.info(f"Loading filter policy model from {source} to {key}")
        if 'torch' in key:
            model = load_torch_model(source,'filter',device=self._config.device)
        else:
            model = load_model(source, key, self._config.use_remote_models)
        self._items[key] = {
                "model": model
        }

    def load_from_config(self, **config: Any) -> None:
        """
        Load one or more filter policy from a configuration

        The format should be:
            key: path_to_model

        :param config: the configuration
        """
        for key, filename in config.items():
            self.load(filename, key)

    def _predict(self, reaction: RetroReaction) -> float:
        if not isinstance(self.selection, str):
            raise PolicyException("No policy selected.")

        model = self[self.selection]["model"]
        prod_fp, rxn_fp = self._reaction_to_fingerprint(reaction, model)
        if hasattr(model,'torchmodel'):
            return model([torch.Tensor(prod_fp), torch.Tensor(rxn_fp)])
        else:
            return model.predict([prod_fp, rxn_fp])[0][0]

    @staticmethod
    def _reaction_to_fingerprint(
        reaction: RetroReaction, model: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        rxn_fp = _make_fingerprint(reaction, model)
        prod_fp = _make_fingerprint(reaction.mol, model)
        return prod_fp, rxn_fp
