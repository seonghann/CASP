""" Module containing helper classes and routines for serialization.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import test.chem

if TYPE_CHECKING:
    from test.utils.type_utils import Optional, Sequence, Dict, Any


class MoleculeSerializer:
    """
    Utility class for serializing molecules

    The id of the molecule to be serialized can be obtained with:

    .. code-block::

        serializer = MoleculeSerializer()
        mol = Molecule(smiles="CCCO")
        idx = serializer[mol]

    which will take care of the serialization of the molecule.
    """

    def __init__(self) -> None:
        self._store: Dict[int, Any] = {}

    def __getitem__(self, mol: Optional[test.chem.Molecule]) -> Optional[int]:
        if mol is None:
            return None

        id_ = id(mol)
        if id_ not in self._store:
            self._add_mol(mol)
        return id_

    @property
    def store(self) -> Dict[int, Any]:
        """Return all serialized molecules as a dictionary"""
        return self._store

    def _add_mol(self, mol: test.chem.Molecule) -> None:
        id_ = id(mol)
        dict_ = {"smiles": mol.smiles, "class": mol.__class__.__name__}
        if isinstance(mol, test.chem.TreeMolecule):
            dict_["parent"] = self[mol.parent]
            dict_["transform"] = mol.transform
        self._store[id_] = dict_


class MoleculeDeserializer:
    """
    Utility class for deserializing molecules.
    The serialized molecules are created upon instantiation of the class.

    The deserialized molecules can be obtained with:

    .. code-block::

        deserializer = MoleculeDeserializer()
        mol = deserializer[idx]

    """

    def __init__(self, store: Dict[int, Any]) -> None:
        self._objects: Dict[int, Any] = {}
        self._create_molecules(store)

    def __getitem__(self, id_: Optional[int]) -> Optional[test.chem.Molecule]:
        if id_ is None:
            return None
        return self._objects[id_]

    def get_tree_molecules(
        self, ids: Sequence[int]
    ) -> Sequence[test.chem.TreeMolecule]:
        """
        Return multiple deserialized tree molecules

        :param ids: the list of IDs to deserialize
        :return: the molecule objects
        """
        objects = []
        for id_ in ids:
            obj = self[id_]
            if obj is None or not isinstance(obj, test.chem.TreeMolecule):
                raise ValueError(f"Failed to deserialize molecule with id {id_}")
            objects.append(obj)
        return objects

    def _create_molecules(self, store: dict) -> None:
        for id_, spec in store.items():
            if isinstance(id_, str):
                id_ = int(id_)

            cls = spec["class"]
            if "parent" in spec:
                spec["parent"] = self[spec["parent"]]

            kwargs = dict(spec)
            del kwargs["class"]
            self._objects[id_] = getattr(test.chem, cls)(**kwargs)
