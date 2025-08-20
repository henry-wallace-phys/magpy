from dataclasses import dataclass
from enum import Enum
from typing import List

import jax.numpy as jnp

class MCEventIndices(Enum):
    TRUE_NEUTRINO_ENERGY = 0
    TRUE_Q2 = 1
    RECO_NEUTRINO_ENERGY = 2
    INTERACTION_MODE = 3
    START_NU = 4
    END_NU = 5
    
    TARGET = 6
    WEIGHT = 7
    NENTRIES = 8
    DUMMY = -1


@dataclass
class MCEvent:
    '''
    Represents a single event in the mc using JAX arrays
    '''
    true_neutrino_energy: float  # True neutrino energy
    true_q2: float  # True Q2
    reco_neutrino_energy: float  # Reconstructed neutrino energy
    interaction_mode: int  # Interaction mode
    target: int  # PDG for target
    
    start_nu: int  # PDG for incoming neutrino
    end_nu: int  # PDG for outgoing neutrino
    weight: float = 0 # Event weight
    
    # Now we want to make it tensor-y
    def to_array(self):
        return jnp.array([
            self.true_neutrino_energy,
            self.true_q2,
            self.reco_neutrino_energy,
            self.interaction_mode,
            self.start_nu,
            self.end_nu,
            self.target,
            self.weight,
        ], dtype=jnp.float64)


class MCEventMonolith:
    '''
    Monolith of MC Events
    '''
    def __init__(self, mc_event_list: List[MCEvent]):
        # Stack all events into a JAX array
        self._mc_event_monolith = jnp.stack(
            [event.to_array() for event in mc_event_list]
        )
        
        self._n_events = len(mc_event_list)

    @property
    def monolith(self):
        return self._mc_event_monolith

    @property
    def n_events(self):
        return self._n_events
    
    def __len__(self):
        return self._n_events


# For backward compatibility
# MCEvent = JAXMCEvent
# MCEventMonolith = JAXMCEventMonolith