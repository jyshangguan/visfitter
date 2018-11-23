import os
import numpy as np
from collections import OrderedDict
from vismodel import *
from srcmodel import *

#-> Dict of the supporting functions
funcLib = {
    "vis_gauss":{
        "x_name": ["u", "v"],
        "param_fit": ["sigma_lm", "A", "i", "pa", "l0", "m0"],
        "param_add": [],
        "operation": ["+"]
    },
    "vis_point":{
        "x_name": ["u", "v"],
        "param_fit": ["A", "l0", "m0"],
        "param_add": [],
        "operation": ["+"]
    },
    "src_gauss":{
        "x_name": ["l", "m"],
        "param_fit": ["sigma_lm", "A", "i", "pa", "l0", "m0"],
        "param_add": [],
        "operation": ["+"]
    }
}
