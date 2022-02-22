# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:59:24 2022

@author: weber.ai
"""
import pickle

class Model_Dict():
    
    def __init__(self):
        pass
    
    def new_dict(self):
        dict = {}
        
        dict.update(
                {"A1": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Output"],
                        "Layer_Nodes": [128, 64, 10],
                        "Layer_Activations": ["Relu", "Relu", "Softmax"],
                        "Optimization": "Adam"
                        }
                    },
                "A2": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Dense", "Dense", 
                                        "Dense", "Output"],
                        "Layer_Nodes": [128, 128, 64, 64, 32, 10],
                        "Layer_Activations": ["Relu", "Relu", "Relu", "Relu", 
                                              "Relu", "Softmax"],
                        "Optimization": "Adam"
                        }
                    },
                "A3": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Dense", "Dense",
                                        "Dense", "Dense", "Dense", "Dense",
                                        "Dense", "Dense", "Dense", "Output"],
                        "Layer_Nodes": [128, 128, 128, 128, 64, 64, 64, 64, 32, 
                                        32, 32, 10],
                        "Layer_Activations": ["Relu", "Relu", "Relu", "Relu",
                                              "Relu", "Relu", "Relu", "Relu", 
                                              "Relu", "Relu", "Relu", "Relu"],
                        "Optimization": "Adam"
                        }
                    },
                "B1": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Output"],
                        "Layer_Nodes": [256, 128, 10],
                        "Layer_Activations": ["Relu", "Relu", "Softmax"],
                        "Optimization": "Adam"
                        }
                    },
                "B2": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Dense", "Dense", 
                                        "Dense", "Output"],
                        "Layer_Nodes": [256, 128, 64, 32, 16, 10],
                        "Layer_Activations": ["Relu", "Relu", "Relu", "Relu", 
                                              "Relu", "Softmax"],
                        "Optimization": "Adam"
                        }
                    },
                "B3": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Dense", "Dense",
                                        "Dense", "Dense", "Dense", "Dense",
                                        "Dense", "Dense", "Dense", "Output"],
                        "Layer_Nodes": [256, 256, 256, 128, 128, 128, 64, 64, 
                                        64, 32, 32, 10],
                        "Layer_Activations": ["Relu", "Relu", "Relu", "Relu",
                                              "Relu", "Relu", "Relu", "Relu", 
                                              "Relu", "Relu", "Relu", "Relu"],
                        "Optimization": "Adam"
                        }
                    },
                "C1": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Output"],
                        "Layer_Nodes": [512, 256, 10],
                        "Layer_Activations": ["Relu", "Relu", "Softmax"],
                        "Optimization": "Adam"
                        }
                    },
                "C2": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Dense", "Dense", 
                                        "Dense", "Output"],
                        "Layer_Nodes": [512, 256, 128, 64, 32, 10],
                        "Layer_Activations": ["Relu", "Relu", "Relu", "Relu", 
                                              "Relu", "Softmax"],
                        "Optimization": "Adam"
                        }
                    },
                "C3": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Dense", "Dense",
                                        "Dense", "Dense", "Dense", "Dense",
                                        "Dense", "Dense", "Dense", "Output"],
                        "Layer_Nodes": [512, 512, 256, 256, 128, 128, 64, 64, 
                                        32, 32, 16, 10],
                        "Layer_Activations": ["Relu", "Relu", "Relu", "Relu",
                                              "Relu", "Relu", "Relu", "Relu", 
                                              "Relu", "Relu", "Relu", "Relu"],
                        "Optimization": "Adam"
                        }
                    },
                "D1": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Output"],
                        "Layer_Nodes": [1024, 512, 10],
                        "Layer_Activations": ["Relu", "Relu", "Softmax"],
                        "Optimization": "Adam"
                        }
                    },
                "D2": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Dense", "Dense", 
                                        "Dense", "Output"],
                        "Layer_Nodes": [1024, 512, 256, 128,  64, 10],
                        "Layer_Activations": ["Relu", "Relu", "Relu", "Relu", 
                                              "Relu", "Softmax"],
                        "Optimization": "Adam"
                        }
                    },
                "D3": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Dense", "Dense", "Dense", "Dense",
                                        "Dense", "Dense", "Dense", "Dense",
                                        "Dense", "Dense", "Dense", "Output"],
                        "Layer_Nodes": [1024, 1024, 512, 512, 256, 256, 128, 
                                        128, 64, 64, 32, 10],
                        "Layer_Activations": ["Relu", "Relu", "Relu", "Relu",
                                              "Relu", "Relu", "Relu", "Relu", 
                                              "Relu", "Relu", "Relu", "Relu"],
                        "Optimization": "Adam"
                        }
                    },
                "E1": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Conv1D", "Conv1D", "MaxPooling1D", 
                                        "Flatten", "Dense", "Output"],
                        "Layer_Nodes": [(64, 4), (64, 4), 0, 0, 128, 10], 
                        "Layer_Activations": ["relu", "relu", 0, 0, "relu", 
                                              "softmax"],
                        "Optimization": "adam"
                        }
                    },
                "E2": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Conv1D", "Conv1D", "MaxPooling1D", 
                                        "Flatten", "Dense", "Output"],
                        "Layer_Nodes": [(128, 4), (128, 4), 0, 0, 256, 10],
                        "Layer_Activations": ["relu", "relu", 0, 0, "relu", 
                                              "softmax"],
                        "Optimization": "adam"
                        }
                    },
                "E3": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Conv1D", "Conv1D", "MaxPooling1D", 
                                        "Flatten", "Dense", "Dense", "Output"],
                        "Layer_Nodes": [(128, 4), (128, 4), 0, 0, 256, 64, 10],
                        "Layer_Activations": ["relu", "relu", 0, 0, "relu", 
                                              "relu", "softmax"],
                        "Optimization": "adam"
                        }
                    },
                "E4": {
                    "Accuracy": [],
                    "Inf_time": [],
                    "Arch": {
                        "Layer_Types": ["Conv1D", "Conv1D", "MaxPooling1D", 
                                        "Flatten", "Dense", "Dense", "Output"],
                        "Layer_Nodes": [(256, 4), (256, 4), 0, 0, 512, 128, 10],
                        "Layer_Activations": ["relu", "relu", 0, 0, "relu", 
                                              "relu", "softmax"],
                        "Optimization": "adam"
                        }
                    }
                })
        return dict
            
            
if __name__ == "__main__":
    MD = Model_Dict()
    model_dict = MD.new_dict()
    with open("model_dict.p", "wb") as f:
        pickle.dump(model_dict, f)
    