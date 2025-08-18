'''
Collection of ROOT file handling classes
'''


from typing import Any

from numpy.typing import NDArray
import pandas as pd 
import torch
import uproot

from magpy.Exceptions import (MagpyFileNotFoundError,
                              MagpyInvalidObjectError)
import logging


class RootFile:
    '''
    Small wrapper around ROOT files
    '''
    
    def __init__(self, file_name: str) -> None:
        try:
            self.open(file_name)
        except Exception as e:
            raise MagpyFileNotFoundError(f"Failed to open MC file: {file_name}") from e

        if self.file is None:
            raise MagpyInvalidObjectError(f"File {file_name} could not be opened or is empty.")


    def open(self, file_name: str) -> None:
        logging.info("Opening ROOT file: [bold]%s", file_name)
        self.file_name = file_name
        self.file = uproot.open(file_name)

        
    def close(self):
        if self.file:
            self.file.close()
            logging.info(f"Closed ROOT file: [bold]{self.file_name}")
        logging.warning("File [bold]%s[/] was already closed or not opened.", self.file_name)

    def get_root_object(self, object_name: str) -> Any:
        '''Get a ROOT object by name'''
        if self.file is None:
            
            raise MagpyInvalidObjectError("File is not opened.")
        try:
            return self.file.get(object_name)
        except Exception as e:
            raise MagpyInvalidObjectError(f"Failed to get ROOT object '{object_name}': {e}") from e
        
class TTreeFile(RootFile):
    '''Interface for files containing TTrees'''
    def __init__(self, file_name: str, tree_name: str = "mc_tree") -> None:
        super().__init__(file_name)
        self.mc_tree = self.get_root_object(tree_name)

        if not isinstance(self.mc_tree, uproot.TTree):
            raise MagpyInvalidObjectError(f"ROOT object '{tree_name}' is not a TTree.")

    def get_tree(self) -> uproot.TTree:
        '''Returns the TTree object from the opened file.'''
        if self.file is None:
            raise MagpyInvalidObjectError("File is not opened.")
        return self.mc_tree
    
    def to_numpy(self)-> NDArray:
        '''Get tree as numpy'''
        return self.mc_tree.arrays(library="np")
    
    def to_pandas(self)->pd.DataFrame:
        '''Get tree as pandas DataFrame'''
        return self.mc_tree.arrays(library="pd")
    
    def to_torch(self)->torch.Tensor:
        '''Get tree as torch Tensor'''
        return torch.tensor(self.to_numpy())

class IterableFile(RootFile):
    '''Interface for files that can be iterated over'''
    def __init__(self, file_name: str, iter_key: str) -> None:
        super().__init__(file_name)
        
        self.file_keys = self.file.keys(filter_classname=iter_key)

    def __getitem__(self, key: str) -> Any:
        if self.file is None:
            raise MagpyInvalidObjectError("File is not opened.")
        if key not in self.file_keys:
            raise MagpyInvalidObjectError(f"Key '{key}' not found in file keys: {self.file_keys}")
        
        return self.get_root_object(key)

    def __iter__(self):
        for key in self.file_keys:
            yield self.get_root_object(key)
