"""
tf_to_dict.py

function for converting keras model files into streamlined dictionary, maintaining important layer info

"""

import tensorflow as tf
import json
from typing import Dict, Any

def tf_to_dict(
    model: tf.keras.Model,
    ) -> Dict[str,Any]:
    
    """
    convert keras model files into streamlined dictionaries, maintaining important layer info

    params
    ------
    model : tf.keras.Model
        a trained and compiled tensorflow model

    returns
    -------
    dict
        a dictionary of important layer info from the tensorflow model
        
    """

    