""" Module containing classes to deal with Molecules and Reactions - mostly wrappers around rdkit routines.
"""
from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdchiral import main as rdc
import torch
from torch.distributions import Categorical

from test.utils.logging import logger
#from test.context.config import Configuration
if TYPE_CHECKING:
    from test.utils.type_utils import (
        Dict,
        Optional,
        Union,
        Tuple,
        List,
        RdMol,
        RdReaction,
        StrDict,
    )


class MoleculeException(Exception):
    """An exception that is raised by the Molecule class"""


class Molecule:
    """
    A base class for molecules. Encapsulate an RDKit mol object and
    functions that can be applied to such a molecule.

    The objects of this class is hashable by the inchi key and hence
    comparable with the equality operator.

    :ivar rd_mol: the RDkit mol object that is encapsulated
    :ivar smiles: the SMILES representation of the molecule

    :param rd_mol: a RDKit mol object to encapsulate, defaults to None
    :param smiles: a SMILES to convert to a molecule object, defaults to None
    :param sanitize: if True, the molecule will be immediately sanitized, defaults to False
    :raises MoleculeException: if neither rd_mol or smiles is given, or if the molecule could not be sanitized
    """

    def __init__(
        self, rd_mol: RdMol = None, smiles: str = None, sanitize: bool = False
    ) -> None:
        if not rd_mol and not smiles:
            raise MoleculeException(
                "Need to provide either a rdkit Mol object or smiles to create a molecule"
            )

        if rd_mol:
            self.rd_mol = rd_mol
            self.smiles = Chem.MolToSmiles(rd_mol)
        else:
            self.smiles = smiles
            self.rd_mol = Chem.MolFromSmiles(smiles, sanitize=False)

        self._inchi_key: Optional[str] = None
        self._inchi: Optional[str] = None
        self._fingerprints: Dict[Union[Tuple[int, int], Tuple[int]], np.ndarray] = {}
        self._is_sanitized: bool = False

        if sanitize:
            self.sanitize()

    def __hash__(self) -> int:
        return hash(self.inchi_key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Molecule):
            return False
        return self.inchi_key == other.inchi_key

    def __len__(self) -> int:
        return self.rd_mol.GetNumAtoms()

    def __str__(self) -> str:
        return self.smiles

    @property
    def inchi(self) -> str:
        """
        The inchi representation of the molecule
        Created by lazy evaluation. Will cause the molecule to be sanitized.

        :return: the inchi
        """
        if not self._inchi:
            self.sanitize(raise_exception=False)
            self._inchi = Chem.MolToInchi(self.rd_mol)
            if self._inchi is None:
                raise MoleculeException("Could not make InChI")
        return self._inchi

    @property
    def inchi_key(self) -> str:
        """
        The inchi key representation of the molecule
        Created by lazy evaluation. Will cause the molecule to be sanitized.

        :return: the inchi key
        """
        if not self._inchi_key:
            self.sanitize(raise_exception=False)
            self._inchi_key = Chem.MolToInchiKey(self.rd_mol)
            if self._inchi_key is None:
                raise MoleculeException("Could not make InChI key")
        return self._inchi_key

    def basic_compare(self, other: "Molecule") -> bool:
        """
        Compare this molecule to another but only to
        the basic part of the inchi key, thereby ignoring stereochemistry etc

        :param other: the molecule to compare to
        :return: True if chemical formula and connectivity is the same
        """
        return self.inchi_key[:14] == other.inchi_key[:14]

    def fingerprint(self, radius: int, nbits: int = 2048) -> np.ndarray:
        """
        Returns the Morgan fingerprint of the molecule

        :param radius: the radius of the fingerprint
        :param nbits: the length of the fingerprint
        :return: the fingerprint
        """
        key = radius, nbits

        if key not in self._fingerprints:
            self.sanitize()
            bitvect = AllChem.GetMorganFingerprintAsBitVect(self.rd_mol, *key)
            array = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(bitvect, array)
            self._fingerprints[key] = array

        return self._fingerprints[key]

    def has_atom_mapping(self) -> bool:
        """
        Determines if a the molecule has atom mappings

        :return: True if at least one atom has a mapping
        """
        for atom in self.rd_mol.GetAtoms():
            if atom.GetAtomMapNum() > 0:
                return True
        return False

    def make_unique(self) -> "UniqueMolecule":
        """
        Returns an instance of the UniqueMolecule class that
        is representing the same molecule but is not hashable or comparable.

        :return: the unique molecule
        """
        return UniqueMolecule(rd_mol=self.rd_mol)

    def remove_atom_mapping(self) -> None:
        """
        Remove all mappings of the atoms and update the smiles
        """
        for atom in self.rd_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        self.smiles = Chem.MolToSmiles(self.rd_mol)

    def sanitize(self, raise_exception: bool = True) -> None:
        """
        Sanitizes the molecule if it has not been done before.

        :param raise_exception: if True will raise exception on failed sanitation
        :raises MoleculeException: if the molecule could not be sanitized
        """
        if self._is_sanitized:
            return

        try:
            AllChem.SanitizeMol(self.rd_mol)
        except:  # noqa, there could be many reasons why the molecule cannot be sanitized
            if raise_exception:
                raise MoleculeException(f"Unable to sanitize molecule ({self.smiles})")
            self.rd_mol = Chem.MolFromSmiles(self.smiles, sanitize=False)
            return

        self.smiles = Chem.MolToSmiles(self.rd_mol)
        self._inchi = None
        self._inchi_key = None
        self._fingerprints = {}
        self._is_sanitized = True


class TreeMolecule(Molecule):
    """
    A special molecule that keeps a reference to a parent molecule.

    If the class is instantiated without specifying the `transform` argument,
    it is computed by increasing the value of the `parent.transform` variable.

    :ivar parent: parent molecule
    :ivar transform: a numerical number corresponding to the depth in the tree

    :param parent: a TreeMolecule object that is the parent
    :param transform: the transform value, defaults to None
    :param rd_mol: a RDKit mol object to encapsulate, defaults to None
    :param smiles: a SMILES to convert to a molecule object, defaults to None
    :param sanitize: if True, the molecule will be immediately sanitized, defaults to False
    :raises MoleculeException: if neither rd_mol or smiles is given, or if the molecule could not be sanitized
    """

    def __init__(
        self,
        parent: Optional["TreeMolecule"],
        transform: int = None,
        rd_mol: RdMol = None,
        smiles: str = None,
        sanitize: bool = False,
    ) -> None:
        super().__init__(rd_mol=rd_mol, smiles=smiles, sanitize=sanitize)
        self.parent = parent
        if transform is None and parent and parent.transform is not None:
            self.transform: int = parent.transform + 1
        else:
            self.transform = transform or 0


class UniqueMolecule(Molecule):
    """
    A special molecule with the hash set to the id of the object.
    Therefore no two instances of this class will be comparable.

    :param rd_mol: a RDKit mol object to encapsulate, defaults to None
    :param smiles: a SMILES to convert to a molecule object, defaults to None
    :param sanitize: if True, the molecule will be immediately sanitized, defaults to False
    :raises MoleculeException: if neither rd_mol or smiles is given, or if the molecule could not be sanitized
    """

    def __init__(
        self, rd_mol: RdMol = None, smiles: str = None, sanitize: bool = False
    ) -> None:
        super().__init__(rd_mol=rd_mol, smiles=smiles, sanitize=sanitize)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, _) -> bool:
        return False


def none_molecule() -> UniqueMolecule:
    """Return an empty molecule"""
    return UniqueMolecule(rd_mol=Chem.MolFromSmiles(""))


#
# Reaction classes below here
#


class _ReactionInterfaceMixin:
    """
    Mixin class to define a common interface for all reaction class

    The methods `_products_getter` and `_reactants_getter` needs to be implemented by subclasses
    """

    def fingerprint(self, radius: int, nbits: int = None) -> np.ndarray:
        """
        Returns a difference fingerprint

        :param radius: the radius of the fingerprint
        :param nbits: the length of the fingerprint. If not given it will use RDKit default, defaults to None
        :return: the fingerprint
        """
        product_fp = sum(
            mol.fingerprint(radius, nbits) for mol in self._products_getter()  # type: ignore
        )
        reactants_fp = sum(
            mol.fingerprint(radius, nbits) for mol in self._reactants_getter()  # type: ignore
        )
        return reactants_fp - product_fp

    def hash_list(self) -> List[str]:
        """
        Return all the products and reactants as hashed SMILES

        :return: the hashes of the SMILES string
        """
        mols = self.reaction_smiles().replace(".", ">>").split(">>")
        return [hashlib.sha224(mol.encode("utf8")).hexdigest() for mol in mols]

    def rd_reaction_from_smiles(self) -> RdReaction:
        """
        The reaction as a RDkit reaction object but created from the reaction smiles
        instead of the SMARTS of the template.

        :return: the reaction object
        """
        return AllChem.ReactionFromSmarts(self.reaction_smiles(), useSmiles=True)

    def reaction_smiles(self) -> str:
        """
        Get the reaction SMILES, i.e. the SMILES of the reactants and products joined together

        :return: the SMILES
        """
        reactants = ".".join(mol.smiles for mol in self._reactants_getter())  # type: ignore
        products = ".".join(mol.smiles for mol in self._products_getter())  # type: ignore
        return "%s>>%s" % (reactants, products)


class _ExpandableReactionInterfaceMixin(_ReactionInterfaceMixin):
    """Mixing for reaction class that has an attribute `smarts` that can be used to create RDKit objects from"""

    @property
    def rd_reaction(self) -> RdReaction:
        """
        The reaction as a RDkit reaction object

        :return: the reaction object
        """
        if not self._rd_reaction:  # type: ignore
            self._rd_reaction = AllChem.ReactionFromSmarts(self.smarts)  # type: ignore
        return self._rd_reaction

    @property
    def smiles(self) -> str:
        """
        The reaction as a SMILES

        :return: the SMILES
        """
        if self._smiles is None:  # type: ignore
            try:
                self._smiles = AllChem.ReactionToSmiles(self.rd_reaction)
            except ValueError:
                self._smiles = ""  # noqa
        return self._smiles


class Reaction(_ExpandableReactionInterfaceMixin):
    """
    An abstraction of a reaction. Encapsulate an RDKit reaction object and
    functions that can be applied to such a reaction.

    :ivar mols: the Molecule objects that this reaction are applied to
    :ivar smarts: the SMARTS representation of the reaction
    :ivar index: a unique index of this reaction,
                 to count for the fact that a reaction can produce more than one outcome
    :ivar metadata: meta data associated with the reaction

    :param mols: the molecules
    :param smarts: the SMARTS fragment
    :param index: the index, defaults to 0
    :param metadata: some meta data
    """

    def __init__(
        self,
        mols: List[Molecule],
        smarts: str,
        index: int = 0,
        metadata: StrDict = None,
    ) -> None:
        self.mols = mols
        self.smarts = smarts
        self.index = index
        self.metadata: StrDict = metadata or {}
        self._products: Optional[Tuple[Tuple[Molecule, ...], ...]] = None
        self._rd_reaction: Optional[RdReaction] = None
        self._smiles: Optional[str] = None
        
    @property
    def products(self) -> Tuple[Tuple[Molecule, ...], ...]:
        """
        Returns the product molecules.
        Apply the reaction if necessary.

        :return: the products of the reaction
        """
        if not self._products:
            self.apply()
            assert self._products is not None
        return self._products

    def apply(self) -> Tuple[Tuple[Molecule, ...], ...]:
        num_rectantant_templates = self.rd_reaction.GetNumReactantTemplates()
        reactants = tuple(mol.rd_mol for mol in self.mols[:num_rectantant_templates])
        products_list = self.rd_reaction.RunReactants(reactants)

        outcomes = []
        for products in products_list:
            try:
                mols = tuple(Molecule(rd_mol=mol, sanitize=True) for mol in products)
            except MoleculeException:
                pass
            else:
                outcomes.append(mols)
        self._products = tuple(outcomes)

        return self._products

    def _products_getter(self) -> Tuple[Molecule, ...]:
        return self.products[self.index]

    def _reactants_getter(self) -> List[Molecule]:
        return self.mols


class RetroReaction_RL(_ExpandableReactionInterfaceMixin):
    """
    A retrosynthesis reaction. Only a single molecule is the reactant.

    :ivar mol: the TreeMolecule object that this reaction is applied to
    :ivar smarts: the SMARTS representation of the reaction
    :ivar index: a unique index of this reaction,
                 to count for the fact that a reaction can produce more than one outcome
    :ivar metadata: meta data associated with the reaction

    :param mol: the molecule
    :param smarts: the SMARTS fragment
    :param index: the index, defaults to 0
    :param metadata: some meta data
    """

    def __init__(
        self, mol: TreeMolecule, 
        smarts: str, 
        index: int = 0, 
        metadata: StrDict = None,
        config: 'Configuration' = None,
        ) -> None:
        self.mol = mol
        self.smarts = smarts
        self.index = index
        self.metadata: StrDict = metadata or {}
        self._reactants: Optional[List[Tuple[TreeMolecule, ...], ...]] = None
        self._rd_reaction: Optional[RdReaction] = None
        self._smiles: Optional[str] = None
        self.config=  config
        self._expanded = False
        self._fingerprints = {}

    @property
    def expanded(self,):
        return self._expanded
    @expanded.setter
    def expanded(self,value):
        self._expanded = value
    
    def get_next_node(self):
        node_idx = self.node_dist.sample()
        node = self.nodes[node_idx]
        return node, node_idx

    def expand(self,new_nodes=None):
        self.expanded = True
        if not self._reactants:
            self.apply()
        if len(self.nodes) != len(self._reactants):
            if new_nodes:
                self.nodes = tuple(new_nodes)
        
        policy = self.config.pair_policy
        node_prob = policy._predict(self)
        self.update_dist(node_prob)
        
    def update_dist(self, node_prob=None):
        if node_prob is not None:
            self.node_prob = node_prob.flatten()

        node_prob = self.node_prob.masked_fill(~self.node_mask, -1e18)
        node_prob = F.softmax(node_prob ,dim = -1) 
        self.node_dist = Categorical(node_prob)

    def del_node(self, node_idx):
        self.node_mask[node_idx] = False
        self.update_dist()

    @property   
    def fingerprint(self,radius: int, nbits: int = None) -> torch.LongTensor: 
        keys = radius, nbits
        if keys in self._fingerprints:
            return self._fingerprints[keys]
        else:
            fp_r = self.mol
            fp_p :Tuple[Tuple[TreeMolecule]] = self.reactants 
            fp_r = sum(
                    mol.fingerprint(radius, nbits) for mol in fp_r
                    )
            fp_p = [sum(mol.fingerprint(radius, nbits) for mol in p) for p in fp_p]
            fp = torch.LongTensor(fp_r) - torch.LongTensor([fp_p])
            self._fingerprints[keys] = fp
            return self._fingerprints[keys]
    
    @classmethod
    def from_reaction_smiles(cls, smiles: str, smarts: str) -> "RetroReaction":
        """
        Construct a retro reaction by parsing a reaction smiles.

        Note that applying reaction does not necessarily give the
        same outcome.

        :param smiles: the reaction smiles
        :param smarts: the SMARTS of the reaction
        :return: the constructed reaction object
        """
        mol_smiles, reactants_smiles = smiles.split(">>")
        mol = TreeMolecule(parent=None, smiles=mol_smiles)
        reaction = RetroReaction(mol, smarts=smarts)
        reaction._reactants = (
            tuple(
                [
                    TreeMolecule(parent=mol, smiles=smiles)
                    for smiles in reactants_smiles.split(".")
                ]
            ),
        )
        return reaction

    def __str__(self) -> str:
        return f"template {self.smarts} on molecule {self.mol.smiles}"

    @property
    def reactants(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        """
        Returns the reactant molecules.
        Apply the reaction if necessary.

        :return: the products of the reaction
        """
        if not self._reactants:
            self._reactants = self.apply()
        return self._reactants

    def apply(self) -> List[Tuple[TreeMolecule, ...], ...]:
        """
        Apply a reactions smarts to a molecule and return the products (reactants for retro templates)

        Will try to sanitize the reactants, and if that fails it will not return that molecule

        :return: the products of the reaction
        """
        reaction = rdc.rdchiralReaction(self.smarts)
        rct = rdc.rdchiralReactants(self.mol.smiles)
        try:
            reactants = rdc.rdchiralRun(reaction, rct)
        except RuntimeError as err:
            logger().debug(
                f"Runtime error in RDChiral with template {self.smarts} on {self.mol.smiles}\n{err}"
            )
            reactants = []

        # Turning rdchiral outcome into rdkit tuple of tuples to maintain compatibility
        outcomes = []
        for reactant_str in reactants:
            smiles_list = reactant_str.split(".")
            try:
                rct = tuple(
                    TreeMolecule(parent=self.mol, smiles=smi, sanitize=True)
                    for smi in smiles_list
                )
            except MoleculeException:
                pass
            else:
                outcomes.append(rct)
        self._reactants = tuple(outcomes)
        if hasattr(self.config,'transition_model'):
            self._set_transition_prob()
        
        self.nodes = tuple()
        return self._reactants

    def copy(self, index: int = None) -> "RetroReaction":
        """
        Shallow copy of this instance.

        :param index: new index, defaults to None
        :return: the copy
        """
        index = index if index is not None else self.index
        new_reaction = RetroReaction(self.mol, self.smarts, index, self.metadata)
        new_reaction._reactants = tuple(mol_list for mol_list in self._reactants or [])
        new_reaction._rd_reaction = self._rd_reaction
        new_reaction._smiles = self._smiles
        return new_reaction
    
    def _set_transition_prob(self,):
        self._transition_prob = list()
        for reactant in self._reactants:
            prob = self.config.transition_model(reactant,self.mol)
            self._transition_prob.append(prob)
        prob = torch.Tensor(self._transition_prob)
        self._transition_dist = Categorical(prob/torch.sum(prob))
    
    def _discard_reactants(self,idx):
        self._reactants.pop(idx)
        self._transition_prob.pop(idx)

    def _check_reactants(self,reactant_idx, check_cycling=True) -> None:
        reactant = self._reactants[reactant_idx]
        if len(reactant) == 1 and self.mol == reactant[0]:
            self._discard_reactants(reactant_idx)
            return False
        
        if check_cycling:
            if self.mol.parent and self.mol.parent in reactant:
                self._discard_reactants(reactant_idx)
                return False
        return True

    def _products_getter(self) -> Tuple[TreeMolecule, ...]:
        return self._reactants[self.index]

    def _reactants_getter(self) -> List[TreeMolecule]:
        return [self.mol]

    def _select(self,):
        idx=self._transition_dist.sample().item()
        if not self._check_reactants(idx):
            prob = torch.Tensor(self._transition_prob)
            self._transition_dist = Categorical(prob/torch.sum(prob))
            return
        else:
            return self._reactants[idx]
    
    def select(self, trial=5):
        for i in range(trial):
            reactant = self._select()
            if reactant:
                return reactant
        return
class RetroReaction(_ExpandableReactionInterfaceMixin):
    """
    A retrosynthesis reaction. Only a single molecule is the reactant.

    :ivar mol: the TreeMolecule object that this reaction is applied to
    :ivar smarts: the SMARTS representation of the reaction
    :ivar index: a unique index of this reaction,
                 to count for the fact that a reaction can produce more than one outcome
    :ivar metadata: meta data associated with the reaction

    :param mol: the molecule
    :param smarts: the SMARTS fragment
    :param index: the index, defaults to 0
    :param metadata: some meta data
    """

    def __init__(
        self, mol: TreeMolecule, 
        smarts: str, 
        index: int = 0, 
        metadata: StrDict = None,
        config: 'Configuration' = None,
        ) -> None:
        self.mol = mol
        self.smarts = smarts
        self.index = index
        self.metadata: StrDict = metadata or {}
        self._reactants: Optional[List[Tuple[TreeMolecule, ...], ...]] = None
        self._rd_reaction: Optional[RdReaction] = None
        self._smiles: Optional[str] = None
        self.config=  config

    @classmethod
    def from_reaction_smiles(cls, smiles: str, smarts: str) -> "RetroReaction":
        """
        Construct a retro reaction by parsing a reaction smiles.

        Note that applying reaction does not necessarily give the
        same outcome.

        :param smiles: the reaction smiles
        :param smarts: the SMARTS of the reaction
        :return: the constructed reaction object
        """
        mol_smiles, reactants_smiles = smiles.split(">>")
        mol = TreeMolecule(parent=None, smiles=mol_smiles)
        reaction = RetroReaction(mol, smarts=smarts)
        reaction._reactants = (
            tuple(
                [
                    TreeMolecule(parent=mol, smiles=smiles)
                    for smiles in reactants_smiles.split(".")
                ]
            ),
        )
        return reaction

    def __str__(self) -> str:
        return f"template {self.smarts} on molecule {self.mol.smiles}"

    @property
    def reactants(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        """
        Returns the reactant molecules.
        Apply the reaction if necessary.

        :return: the products of the reaction
        """
        if not self._reactants:
            self._reactants = self.apply()
        return self._reactants

    def apply(self) -> List[Tuple[TreeMolecule, ...], ...]:
        """
        Apply a reactions smarts to a molecule and return the products (reactants for retro templates)

        Will try to sanitize the reactants, and if that fails it will not return that molecule

        :return: the products of the reaction
        """
        reaction = rdc.rdchiralReaction(self.smarts)
        rct = rdc.rdchiralReactants(self.mol.smiles)
        try:
            reactants = rdc.rdchiralRun(reaction, rct)
        except RuntimeError as err:
            logger().debug(
                f"Runtime error in RDChiral with template {self.smarts} on {self.mol.smiles}\n{err}"
            )
            reactants = []

        # Turning rdchiral outcome into rdkit tuple of tuples to maintain compatibility
        outcomes = []
        for reactant_str in reactants:
            smiles_list = reactant_str.split(".")
            try:
                rct = tuple(
                    TreeMolecule(parent=self.mol, smiles=smi, sanitize=True)
                    for smi in smiles_list
                )
            except MoleculeException:
                pass
            else:
                outcomes.append(rct)
        self._reactants = tuple(outcomes)
        
        if hasattr(self.config,'transition_model'):
            self._set_transition_prob()
        return self._reactants

    def copy(self, index: int = None) -> "RetroReaction":
        """
        Shallow copy of this instance.

        :param index: new index, defaults to None
        :return: the copy
        """
        index = index if index is not None else self.index
        new_reaction = RetroReaction(self.mol, self.smarts, index, self.metadata)
        new_reaction._reactants = tuple(mol_list for mol_list in self._reactants or [])
        new_reaction._rd_reaction = self._rd_reaction
        new_reaction._smiles = self._smiles
        return new_reaction
    
    def _set_transition_prob(self,):
        self._transition_prob = list()
        for reactant in self._reactants:
            prob = self.config.transition_model(reactant,self.mol)
            self._transition_prob.append(prob)
        prob = torch.Tensor(self._transition_prob)
        self._transition_dist = Categorical(prob/torch.sum(prob))
    
    def _discard_reactants(self,idx):
        self._reactants.pop(idx)
        self._transition_prob.pop(idx)

    def _check_reactants(self,reactant_idx, check_cycling=True) -> None:
        reactant = self._reactants[reactant_idx]
        if len(reactant) == 1 and self.mol == reactant[0]:
            self._discard_reactants(reactant_idx)
            return False
        
        if check_cycling:
            if self.mol.parent and self.mol.parent in reactant:
                self._discard_reactants(reactant_idx)
                return False
        return True

    def _products_getter(self) -> Tuple[TreeMolecule, ...]:
        return self._reactants[self.index]

    def _reactants_getter(self) -> List[TreeMolecule]:
        return [self.mol]

    def _select(self,):
        idx=self._transition_dist.sample().item()
        if not self._check_reactants(idx):
            prob = torch.Tensor(self._transition_prob)
            self._transition_dist = Categorical(prob/torch.sum(prob))
            return
        else:
            return self._reactants[idx]
    
    def select(self, trial=5):
        for i in range(trial):
            reactant = self._select()
            if reactant:
                return reactant
        return

class FixedRetroReaction(_ReactionInterfaceMixin):
    """
    A retrosynthesis reaction that has the same interface as `RetroReaction`
    but it is fixed so it does not support SMARTS application.

    The reactants are set by using the `reactants` property.

    :ivar mol: the UniqueMolecule object that this reaction is applied to
    :ivar smiles: the SMILES representation of the RDKit reaction
    :ivar metadata: meta data associated with the reaction
    :ivar reactants: the reactants of this reaction

    :param mol: the molecule
    :param smiles: the SMILES of the reaction
    :param metadata: some meta data
    """

    def __init__(
        self, mol: UniqueMolecule, smiles: str = "", metadata: StrDict = None
    ) -> None:
        self.mol = mol
        self.smiles = smiles
        self.metadata = metadata or {}
        self.reactants: Tuple[Tuple[UniqueMolecule, ...], ...] = ()


    def copy(self) -> "FixedRetroReaction":
        """
        Shallow copy of this instance.

        :return: the copy
        """
        new_reaction = FixedRetroReaction(self.mol, self.smiles, self.metadata)
        new_reaction.reactants = tuple(mol_list for mol_list in self.reactants)
        return new_reaction

    def _products_getter(self) -> Tuple[UniqueMolecule, ...]:
        return self.reactants[0]

    def _reactants_getter(self) -> List[UniqueMolecule]:
        return [self.mol]


def hash_reactions(
    reactions: Union[List[Reaction], List[RetroReaction], List[FixedRetroReaction]],
    sort: bool = True,
) -> str:
    """
    Creates a hash for a list of reactions

    :param reactions: the reactions to hash
    :param sort: if True will sort all molecules, defaults to True
    :return: the hash string
    """
    hash_list = []
    for reaction in reactions:
        hash_list.extend(reaction.hash_list())
    if sort:
        hash_list.sort()
    hash_list_str = ".".join(hash_list)
    return hashlib.sha224(hash_list_str.encode("utf8")).hexdigest()
