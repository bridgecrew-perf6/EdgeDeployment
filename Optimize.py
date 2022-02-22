# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 11:36:23 2022

@author: weber.ai
"""

import pickle

class optimize():
    
    def __init__(self):
        with open("model_dict2.p", "rb") as f:
            self.model_dict = pickle.load(f)
    
    def opt_without_quant(self):
        pass
    
    def opt_with_quant(self):
        pass
    
    def save_models(self):
        with open("model_dict3.p", "wb") as f:
            pickle.dump(self.model_dict, f)
            
if __name__ == "__main__":
    Opt = optimize()
    Opt.save_models()