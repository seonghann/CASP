import pickle
import numpy
from typing import TYPE_CHECKING

from type_utils import (
        StrDict, 
        Set, 
        Union, 
        Any, 
        Optional, 
        List, 
        RdMol, 
        RetroRxn,
        Tuple,
        )
import random
class EnvCASP():

    def __init__(self,):

        self.action_generator = None
        self.root = None
        self.stock = None
        self.reward_manager = None
        
    def step(self,action: (RetroRxn,int)) -> (State, float, bool):
        idx = self.select_mol()
        new_state = action.apply(self.current_state,idx)
        
        reward = self.get_reward(self.current_state, action)
        self.current_state = new_state
        done = self.check_done()
        return new_state, reward, done

    def select_mol(self,):
        return random.randrange(0,len(self.current_state))

    def get_reward(self,):
        if self.reward_manager:
            return 1
        else:
            return 1

    def check_in_stock(self,mol: RdMol) -> bool:
        return mol in self.stock

    def check_done(self,) -> bool:
        l = []
        for mol in self.current_state.mols:
            in_stock = self.check_in_stock(mol)
            if not in_stock:
                return in_stock
        return True


