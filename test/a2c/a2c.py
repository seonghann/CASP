""" Module containing a class that holds the tree search
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING

import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

from test.a2c.test2 import Node
from test.utils.serialization import MoleculeSerializer, MoleculeDeserializer

if TYPE_CHECKING:
    from test.utils.type_utils import Tuple, List, Optional
    from test.a2c.config import Configuration
    from test.chem import RetroReaction_RL as RetroReaction


class SearchTree:
    """
    Encapsulation of the search tree.

    :ivar root: the root node
    :ivar config: the configuration of the search tree

    :param config: settings of the tree search algorithm
    :param root_smiles: the root will be set to a node representing this molecule, defaults to None
    """

    def __init__(self, config: Configuration, root_smiles: str = None) -> None:
        if root_smiles:
            self.root: Optional[Node] = Node.create_root(
                smiles=root_smiles, tree=self, config=config
            )
        else:
            self.root = None
        self.config = config
        self._graph: Optional[nx.DiGraph] = None
        
        self.policy_template = config.expansion_policy


    @classmethod
    def from_json(cls, filename: str, config: Configuration) -> "SearchTree":
        """
        Create a new search tree by deserialization from a JSON file

        :param filename: the path to the JSON node
        :param config: the configuration of the search
        :return: a deserialized tree
        """
        tree = SearchTree(config)
        with open(filename, "r") as fileobj:
            dict_ = json.load(fileobj)
        mol_deser = MoleculeDeserializer(dict_["molecules"])
        tree.root = Node.from_dict(dict_["tree"], tree, config, mol_deser)
        return tree


    def backpropagate(self, from_node: Node, value_estimate: float) -> None:
        """
        Backpropagate the value estimate and update all nodes from a
        given node all the way to the root.

        :param from_node: the end node of the route to update
        :param value_estimate: The score value to backpropagate
        """
        current = from_node
        while current is not self.root:
            parent = current.parent
            parent.backpropagate(current, value_estimate)
            current = parent

    def graph(self, recreate: bool = False) -> nx.DiGraph:
        """
        Construct a directed graph object with the nodes as
        vertices and the actions as edges attribute "action".

        :param recreate: if True will construct the graph even though it is cached, defaults to False
        :return: the graph object
        :raises ValueError: if the tree is not defined
        """
        if not self.root:
            raise ValueError("Root of search tree is not defined ")

        if not recreate and self._graph:
            return self._graph

        def add_node(node):
            self._graph.add_edge(node.parent, node, action=node.parent[node]["action"])
            for grandchild in node.children():
                add_node(grandchild)

        self._graph = nx.DiGraph()
        # Always add the root
        self._graph.add_node(self.root)
        for child in self.root.children():
            add_node(child)
        return self._graph

    def select_leaf(self) -> Node:
        """
        Traverse the tree selecting the most promising child at
        each step until leaf node returned.

        :return: the leaf node
        :raises ValueError: if the tree is not defined
        """
        if not self.root:
            raise ValueError("Root of search tree is not defined ")

        current = self.root
        while current.is_expanded and not current.state.is_solved:
            promising_child, action_key, action_idx, child_idx =\
                                                self.promising_child(current)
            # If promising_child returns None it means that the node
            # is unexpandable, and hence we should break the loop
            if promising_child:
                current = promising_child
        return current

    def promising_child(self,node: Node) -> Optional["Node"]:
        """
        Return the child with the currently highest Q+U

        The selected child will be instantiated if it has not been already.

        If no actions could be found that were applicable, the method will
        return None.

        :return: the child
        """
        if not node.is_expandable:
            return None, None, None, None

        action_key, action_idx = self.select_action(node)
        if action_key not in node.action.keys():
            node.add_action(action_key)
        
        child, child_idx = self.select_child(node,action_key) # need to check
        # action.select_child
        if not child and node.is_expandable:
            #node.action[action_key].remove_child(child_idx)
            return self.promising_child(node)

        if not child:
            self._logger.debug(
                "Returning None from promising_child() because there were no applicable action"
            )
            node.is_expanded = False
            node.is_expandable = False

        return child, action_key, action_idx, child_idx

    def select_action(self, node: Node):
        
        if not node.is_expanded and node.is_expandable:
            self.expand(node)

        action_number = node.action_dist.sample()
        num_templates = node.action_mask.size(1)
        key = (action_number.item()//num_templates, 
               action_number.item()%num_templates)
        return key, action_number

    def select_child(self, node: Node, action_key: tuple) -> Optional["Node"]:
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
        
        action = node.action[action_key]

        if not action.expanded:
            if not action.reactants:
                action.apply()
                if len(action.reactants) == 0:
                    node.del_action(action_key)
                    return None, None

                new_nodes = node.create_children_nodes(action_key)
                action.expand(new_nodes=new_nodes)
            else:
                action.expand()

        next_node, next_node_idx = action.get_next_node()

        if self._config.prune_cycles_in_search:
            if not node.check_child_node(action_key, next_node_idx):
                action.del_node(next_node_idx)
                return None,None

        if hasattr(action,'node_mask') and not any(action.node_mask):
            node.del_action(action_key)
            return None,None

        return next_node, next_node_idx
    
    def masking_action(self, node: Node) -> None:
        """from template library, get applicability
        """
        action_mask = list()
        self.policy_template.select_first()
        policy_key = self.policy_template.selection[0]
        templates = self.policy_template[policy_key]\
                        ['templates']['rdChemReaction'].values

        for mol in node.state.mols:
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

        node.context_mask = torch.stack(action_mask)
        node.masking_action = True
    
    def masking_action2(self, node: Node) -> None:
        
        self.template_masking_model.select_first()
        mask = self.template_masking_model.get_mask(node.state.mols)
        mask = mask > self.config.template_mask_cutoff

        for i,mol in enumerate(node.state.mols):
            if mol in self._config.stock:
                mask[i] = False
        
        node.context_mask = mask
        node.masking_action = True
        
    def expand(self, node: Node) -> None:
        """
        Expand the node.

        Expansion is the process of creating the children of the node,
        without instantiating a child object. The actions and priors are
        taken from the policy network.
        """
        if node.is_expanded:
            msg = f"Oh no! This node is already expanded. id={id(node)}"
            self._logger.debug(msg)
            #raise NodeUnexpectedBehaviourException(msg)

        if node.is_expanded or not node.is_expandable:
            return

        node.is_expanded = True
        
        # Calculate the template is applicable.
        if not node.masking_action:
            self.masking_action(node)

        possible_action, action_raw_prob, cutoff_mask = \
                self.expansion_policy.get_actions(node.state.mols,
                                                   node.context_mask) 
        node.cutoff_mask = cutoff_mask
        node.action_mask = node.cutoff_mask & node.context_mask
        if not any(node.action_mask.flatten()):
            node.is_expandable = False
            return
        else:
            node.update_actions(action_raw_prob, actions=possible_action)

    def serialize(self, filename: str) -> None:
        """
        Serialize the search tree to a JSON file

        :param filename: the path to the JSON file
        :raises ValueError: if the tree is not defined
        """
        if not self.root:
            raise ValueError("Root of search tree is not defined ")

        mol_ser = MoleculeSerializer()
        dict_ = {"tree": self.root.serialize(mol_ser), "molecules": mol_ser.store}
        with open(filename, "w") as fileobj:
            json.dump(dict_, fileobj, indent=2)

    @staticmethod
    def route_to_node(from_node: Node) -> Tuple[List[RetroReaction], List[Node]]:
        """
        Return the route to a give node to the root.

        Will return both the actions taken to go between the nodes,
        and the nodes in the route themselves.

        :param from_node: the end of the route
        :return: the route
        """
        actions = []
        nodes = []
        current = from_node

        while current is not None:
            parent = current.parent
            if parent is not None:
                action = parent[current]["action"]
                actions.append(action)
            nodes.append(current)
            current = parent
        return actions[::-1], nodes[::-1]
