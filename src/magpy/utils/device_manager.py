import torch


class DeviceManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)

    def from_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to("cpu")
