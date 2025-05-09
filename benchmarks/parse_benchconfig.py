class Benchconfig:

    def __init__(self,
                 path: str,
                 ):
        self.config_path = path
        self.load_config()

    """
    Create a dict for the config file:
        - if you want to understand how config files need to be written have a look at the one I wrote (comment)
        - The dict has entries of the form: key = f"{layer}_{parameter}" and value = List[values with space separation]
    """
    def load_config(self):
        self.config = {}
        layer= "invalid_layer"
        with open(self.config_path, 'r') as f:
            for line in f:
                line = line.strip() #remove whitespace
                if line and not line.startswith('#'):  #Ignore comments and empty lines
                    if ':' in line:
                        key, value = line.split(':', 1) #: used a key value separator
                        if(key.strip() == "layer"): #if layer key we change the layer var changes the meaning of following key value pairs
                            layer = value.strip()
                        else: #if we have an actual config key value pair add it to the dict
                            self.config[f"{layer}_{key.strip()}"] = [elem for elem in value.strip().split()]
                    else:
                        print(f"Parsing error of {self.config_path}, non empty, non comment line has no : key value separator")
