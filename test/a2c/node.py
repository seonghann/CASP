""" Module containing a class that represents a node in the search tree.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import pickle
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions import Categorical
import numpy as np

from test.chem import RetroReaction, TreeMolecule
from test.mcts.state import State
from test.utils.logging import logger

if TYPE_CHECKING:
    from test.utils.type_utils import StrDict, List, Optional, Tensor
    from test.utils.serialization import (
        MoleculeDeserializer,
        MoleculeSerializer,
    )
    from test.context.config import Configuration
    from test.mcts.mcts import SearchTree


class NodeUnexpectedBehaviourException(Exception):
    """Exception that is raised if the tree search is behaving unexpectedly."""
class NodeExpandError(Exception):
    """Exception from expand a node which is already expanded"""


class Node:
    """
    A node in the search tree.

    The children are instantiated lazily for efficiency: only when
    a child is selected the reaction to create that child is applied.

    Properties of an instantiated children to a node can be access with:

    .. code-block::

        children_attr = node[child]

    the return value is a dictionary with keys "action", "value", "prior"
    and "visitations".

    :ivar is_expanded: if the node has had children added to it
    :ivar is_expandable: if the node is expandable
    :ivar is_applicable: if the node has been checked is applicable
    
    :ivar parent: the parent node
    :ivar parent_action: the action from which itself comes out
    
    :ivar state: the internal state of the node
    :ivar tree: the tree owning this node
    
    :ivar _action: dict of class <RetroReaction>, where key is (i,j); i_th molecules and j_th template 
    :ivar _action_mask: torch.BoolTensor , n_mol x n_templates possible action is True.
    :ivar _action_raw_prob: torch.longTensor size([n_mol,n_templates])
    :ivar _action_prob: 1) masking _action_raw_prob 2) softmax 3) flatten() 4) normalize
    :ivar _action_dist: Categorical(_action_prob)

    :param state: the state of the node
    :param owner: the tree that owns this node
    :param config: settings of the tree search algorithm
    :param parent: the parent node, defaults to None
    """

    def __init__(
        self, state: State, owner: SearchTree, config: Configuration, parent=None, parent_action=None
    ):
        self.state = state
        self._config = config
        self._expansion_policy = config.expansion_policy # policy -> _expansion_policy 
        
        self.tree = owner       
        self.is_expanded: bool = False                   
        self.is_applicable: bool = False
        self.is_expandable: bool = not self.state.is_terminal
        
        self.parent = parent
        self.parent_action = parent_action
        
        self._action: dict[RetroReaction] 
        self._action_mask: torch.Tensor 
        self._action_raw_prob: torch.Tensor

        self._logger = logger()

    def __getitem__(self, action_key: tuple or int) -> StrDict:
        if action_key in self._action.keys():
            return self._action[action_key]

        elif isinstance(action_key,int) and action_key<len(self.state.mols):
            actions = list()
            for k in self._action.keys():
                if k[0] == action_key:
                    actions.append(self._action[k])
            return actions
        
        else:
            raise

    @classmethod
    def create_root(
        cls, smiles: str, tree: SearchTree, config: Configuration
    ) -> "Node":
        """
        Create a root node for a tree using a SMILES.

        :param smiles: the SMILES representation of the root state
        :param tree: the search tree
        :param config: settings of the tree search algorithm
        :return: the created node
        """
        mol = TreeMolecule(parent=None, transform=0, smiles=smiles)
        state = State(mols=[mol], config=config)
        return Node(state=state, owner=tree, config=config)

    @classmethod
    def from_dict(
        cls,
        dict_: StrDict,
        tree: SearchTree,
        config: Configuration,
        molecules: MoleculeDeserializer,
        parent: "Node" = None,
    ) -> "Node":
        """
        Create a new node from a dictionary, i.e. deserialization

        :param dict_: the serialized node
        :param tree: the search tree
        :param config: settings of the tree search algorithm
        :param molecules: the deserialized molecules
        :param parent: the parent node
        :return: a deserialized node
        """
        state = State.from_dict(dict_["state"], config, molecules)
        node = Node(state=state, owner=tree, config=config, parent=parent)
        node.is_expanded = dict_["is_expanded"]
        node.is_expandable = dict_["is_expandable"]
        node._children_values = dict_["children_values"]
        node._children_priors = dict_["children_priors"]
        node._children_visitations = dict_["children_visitations"]
        node._children_actions = []
        for action in dict_["children_actions"]:
            mol = molecules.get_tree_molecules([action["mol"]])[0]
            node._children_actions.append(
                RetroReaction(
                    mol,
                    action["smarts"],
                    action["index"],
                    action.get("metadata", {}),
                )
            )

        node._children = [
            Node.from_dict(child, tree, config, molecules, parent=node)
            if child
            else None
            for child in dict_["children"]
        ]
        return node
    
    def view(self) -> StrDict:
        """
        Creates a view of the children attributes. Each of the
        list returned is a new list, although the actual children
        are not copied.

        The return dictionary will have keys "actions", "values",
        "priors", "visitations" and "objects".

        :return: the view
        """
        return {
                "actions": list(self._action),
                "values": list(self._action_prob),
        }

    def expand(self) -> None:
        """
        Expand the node.

        Expansion is the process of creating the action space.
        Object is created : 
                self._action_mask, 
                self._action_raw_prob, 
                self._action_raw_prob, 
                self._action_prob
        The actions are generated from template application.
        The probability are taken from the policy network.
        """
        if self.is_expanded:
            msg = f"Oh no! This node is already expanded. id={id(self)}"
            self._logger.debug(msg)
            raise NodeExpandError(msg)

        if self.is_expanded or not self.is_expandable:
            msg = f'The node is not expandable. id={id(self)}'
            self._logger.debug(msg)
            return

        self.is_expanded = True
        
        # Calculate the template is applicable.
        if not self.is_applicable:
            self._check_applicability()

        possible_action, self._action_raw_prob = \
                self._expansion_policy.get_actions2( 
                        self.state.mols, self._action_mask) 

        if not any(self._action_mask.flatten()):
            self.is_expandable = False
            return
        else:
            self._get_action_prob()
            self._action_dist = Categorical(self._action_prob)
            self._action = possible_action
    
    def _get_action_prob(self,):
        print('=======================')
        print('Node._get_action_prob()')
        self._action_prob = F.softmax(
                            self._action_raw_prob.masked_fill(
                            ~self._action_mask,-1e18), 
                            dim=-1
                            )
        self._action_prob = self._action_prob.masked_fill(~self._action_mask,0)
        self._action_prob = self._action_prob.flatten()
        self._action_prob /= torch.sum(self._action_prob)
        print(f'action_raw_prob : {self._action_raw_prob.shape},\n'
              f'action_raw_prob : {self._action_raw_prob},\n'
              f'action_mask : {self._action_mask.shape},\n'
              f'action_mask : {self._action_mask},\n'
              f'action_prob : {self._action_prob.shape},\n'
              f'action_prob : {self._action_prob}')
        print('=======================')
        return self._action_prob

    def _initialize_action_prob(self,):
        self._action_raw_prob = torch.zeros(self._action_raw_prob.size())
        self._action_prob = torch.zeros(self._action_prob.size())
        self._action_dist = None

    def _check_applicability(self) -> None:
        """from template library, get applicability
        """
        action_mask = list()
        self._expansion_policy.select_first()
        policy_key = self._expansion_policy.selection[0]
        templates = self._expansion_policy[policy_key]\
                        ['templates']['rdChemReaction'].values

        for mol in self.state.mols:
            if mol in self._config.stock:
                ith_mol_action_mask = [False] * len(templates)
                action_mask.append(torch.Tensor(ith_mol_action_mask).bool())
            else:
                if mol.rd_mol:
                    mol = mol.rd_mol
                else:
                    mol = Chem.MolFromSmiles(mol.smiles)
                ith_mol_action_mask = [False] * len(templates)
                for idx, template in enumerate(templates):
                    R = template.GetReactantTemplate(0)
                    if mol.GetSubstructMatch(R):
                        ith_mol_action_mask[idx] = True
                action_mask.append(torch.Tensor(ith_mol_action_mask).bool())
        self._action_mask = torch.stack(action_mask)
        self.is_applicable = True


    def is_terminal(self) -> bool:
        """
        Node is terminal if its unexpandable, or the internal state is terminal (solved)

        :return: the terminal attribute of the node
        """
        return not self.is_expandable or self.state.is_terminal

    def _select_action(self):
        action_number = self._action_dist.sample()
        num_templates = self._action_mask.size(1)
        key = (action_number.item()//num_templates, 
               action_number.item()%num_templates)
        return key, action_number

    def promising_child(self,action_key_=None,node_idx_=None) -> Optional["Node"]:
        """
        Return the child with the currently highest Q+U

        The selected child will be instantiated if it has not been already.

        If no actions could be found that were applicable, the method will
        return None.

        :return: the child
        """
        if not self.is_expandable:
            return None, None, None, None

        action_key, action_idx = self._select_action()
        child, node_idx = self._select_child(action_key)
        if action_key_:
            child, node_idx = self._select_child(action_key_,node_idx_=node_idx_)
            action_key = action_key_ 
            node_idx = node_idx_

        if not child and self._action:
            return self.promising_child()

        if not child:
            self._logger.debug(
                "Returning None from promising_child() because there were no applicable action"
            )
            self.is_expanded = False
            self.is_expandable = False

        return child, action_key, action_idx, node_idx
    
    def _select_child(self, action_key: tuple, node_idx_=None) -> Optional["Node"]:
        """
        Selecting a child node implies instantiating the children nodes

        The algorithm is:
        * If the child has already been instantiated, return immediately
        * Apply the reaction associated with the child
        * If the application of the action failed, set value to -1e6 and return None
        * Create a new state array, one new state for each of the reaction outcomes
        * Create new child nodes
            - If a filter policy is available and the reaction outcome is unlikely
              set value of child to -1e6
        * Select a random node of the feasible ones to return
        """
        
        action = self._action[action_key]
        if action.expanded:
            return action.get_next_node()

        if not action._reactants:
            action.apply()

        keep_mols = [
            mol for i,mol in enumerate(self.state.mols) if i != action_key[0]
            ]
        new_states = [
            State(keep_mols + list(reactants), self._config)
            for reactants in action.reactants
            ]
        new_nodes = self._create_children_nodes(new_states, action)
        action.add_nodes(new_nodes)

        #print('===================')
        #tmp = [node.state.mols for node in action.next_node]
        #for i in tmp:
        #    msg = ''
        #    for mol in i:
        #        msg+=mol.smiles+' . '
        #    print(msg)
        #print('===================')
        next_node_idx = -1
        if hasattr(action,'node_mask') and not any(action.node_mask):
            self._del_action(action_key)
            return None,None
        
        next_node, next_node_idx = action.get_next_node()

        if self._config.prune_cycles_in_search:
            if not self._check_child_node(action_key, next_node_idx):
                action.del_node(next_node_idx)
                return None,None

        return next_node, next_node_idx

    def _del_action(self,action_key):
        self._action_mask[action_key] = False
        del(self._action[action_key])
        if not any(self._action_mask.faltten()):
            self.is_expandable = False
            return
        else:
            self._get_action_prob()
            self._action_dist = Categorical(self._action_prob)

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        Serialize the node object to a dictionary

        :param molecule_store: the serialized molecules
        :return: the serialized node
        """
        return {
            "state": self.state.serialize(molecule_store),
            "children_values": [float(value) for value in self._children_values],
            "children_priors": [float(value) for value in self._children_priors],
            "children_visitations": self._children_visitations,
            "children_actions": [
                {
                    "mol": molecule_store[action.mol],
                    "smarts": action.smarts,
                    "index": action.index,
                    "metadata": dict(action.metadata),
                }
                for action in self._children_actions
            ],
            "children": [
                child.serialize(molecule_store) if child else None
                for child in self._children
            ],
            "is_expanded": self.is_expanded,
            "is_expandable": self.is_expandable,
        }

    def _check_child_node(self, action_key: tuple, node_idx: int) -> bool:
        action = self._action[action_key]
        reactant = action.reactants[node_idx]
        product = action.mol
        if product in reactant:
            return False
        if hasattr(product,'parent'):
            if product.parent in reactant:
                return False
        return True

    def _create_children_nodes(
            self, states: List[State], action: RetroReaction=None
    ) -> List[Optional["Node"]]:
        new_nodes = []
        for state in states:
            node = Node(state=state, owner=self.tree, 
                    config=self._config, parent=self, parent_action=action)
            new_nodes.append(node)
        return new_nodes

class _(nn.Module):
    def __init__(self,):
        super(_, self).__init__()
    def forward(self,x):
        return torch.zeros(121) + 1/121
