import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
import model.questions as mm
from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    try: 
        import numpy as np
        import pandas as pd
        import torch
        import sklearn
        import fastai
    except:
        return "Install necessary packages"
    
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        response = mm.start(text)

        output.append(response)

    return SimpleText(dict(text=output))
