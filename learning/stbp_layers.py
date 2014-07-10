#!/usr/bin/env python 

from __future__ import division

"""
This file exist for backward compatibility and presets various 
classes under their old name:

    STBPSTack = LayerStack

    FactoizedBernoulliTop, SigmoidBeliefLayer = SBNTop, SBN
    NADE, CNADE = NADETop, NADE

"""


from rws import *

from sbn import SBN, SBNTop
from darn import DARN, DARNTop
from nade import NADE, NADETop

STBPStack = LayerStack

SigmoidBeliefLayer, FactoizedBernoulliTop = SBN, SBNTop
CNADE, NADE = NADE, NADETop

