#!/usr/bin/env python 

from __future__ import division

"""
This file exist for backward compatibility and presets various 
classes under their old name:

    STBPSTack = ISStack

    FactoizedBernoulliTop, SigmoidBeliefLayer = SBNTop, SBN
    NADE, CNADE = NADETop, NADE

"""


from isws import *

from sbn import SBN, SBNTop
from darn import DARN, DARNTop
from nade import NADE, NADETop

STBPStack = ISStack

SigmoidBeliefLayer, FactoizedBernoulliTop = SBN, SBNTop
CNADE, NADE = NADE, NADETop
