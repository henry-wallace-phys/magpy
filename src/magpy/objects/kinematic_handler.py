from abc import ABC, abstractmethod
from jax import numpy as jnp
from magpy.objects.mc_event import MCEventMonolith, MCEventIndices


class Kinematic(ABC):
    """
    Interface for kinematic evaluations.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def evaluate(self, event: MCEventMonolith) -> jnp.array: ...


# ------- Specific kinematics ---------


class TrueNeutrinoEnergy:
    def evaluate(self, event: MCEventMonolith) -> jnp.array:
        """Get the true neutrino energy"""
        return event.monolith[:, MCEventIndices.TRUE_NEUTRINO_ENERGY.value]


class TrueQ2:
    def evaluate(self, event: MCEventMonolith) -> jnp.array:
        """Get the true Q2"""
        return event.monolith[:, MCEventIndices.TRUE_Q2.value]


class RecoNeutrinoEnergy:
    def evaluate(self, event: MCEventMonolith) -> jnp.array:
        """Get the reconstructed neutrino energy"""
        return event.monolith[:, MCEventIndices.RECO_NEUTRINO_ENERGY.value]


class InteractionMode:
    def evaluate(self, event: MCEventMonolith) -> jnp.array:
        """Get the interaction mode"""
        return event.monolith[:, MCEventIndices.INTERACTION_MODE.value]


class OscChannel:
    """Get the oscillation channel"""

    def evaluate(self, event: MCEventMonolith) -> jnp.array:
        return event.monolith[
            :,
            jnp.array([MCEventIndices.START_NU.value, MCEventIndices.END_NU.value]),
        ]


# ----------------
class KinematicFactory:
    """ """

    @staticmethod
    def create_kinematic(name: str) -> Kinematic:
        """
        Create a kinematic object based on the name.
        """
        kinematic_classes = {
            "true_neutrino_energy": TrueNeutrinoEnergy,
            "true_q2": TrueQ2,
            "reco_neutrino_energy": RecoNeutrinoEnergy,
            "interaction_mode": InteractionMode,
            "osc_channel": OscChannel,
        }
        return kinematic_classes.get(name, lambda: None)()


# --------------
# Binning
class KinematicBinning:
    def __init__(self, variable: str, binning: jnp.array):
        self._handler = KinematicFactory.create_kinematic(variable)
        self.binning = binning

    def get_bin(self, event: MCEventMonolith) -> jnp.array:
        """Get the bin index for a given value"""
        kinematic_value = self._handler.evaluate(event)
        return torch.searchsorted(self.binning, kinematic_value)
