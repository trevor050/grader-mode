"""Apply the frozen detector to an ordered 60-coordinate activation vector."""
import json
from pathlib import Path
import numpy as np


def load(path=None):
    return json.loads(Path(path or Path(__file__).with_name('detector.json')).read_text())


def score(features, model):
    x=np.asarray(features,dtype=np.float64)
    if x.shape[-1:]!=(60,) or not np.isfinite(x).all():
        raise ValueError('Expected finite activations with last dimension 60')
    return np.sum(((x-np.asarray(model['mean']))/np.asarray(model['sd']))*np.asarray(model['coef']),axis=-1)+model['intercept']


def fires(features,model):
    return score(features,model)>=model['threshold']
