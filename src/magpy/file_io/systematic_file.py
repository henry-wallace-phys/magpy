import yaml

from magpy.objects.systematic_handler import SystematicHandler, Systematic

class SystematicFile:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as file:
            self.data = yaml.safe_load(file)
        
        syst_list = ['']*len(self.data['Systematics'])    
        
        for i, entry in enumerate(self.data['Systematics']):
            if not isinstance(entry, dict):
                raise ValueError("Each entry in the YAML file must be a dictionary.")
            syst_list[i] = Systematic(**entry)
            
        self._systematic_handler = SystematicHandler(syst_list)
    
    @property 
    def systematic_handler(self)->SystematicHandler:
        return self._systematic_handler