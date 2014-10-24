
from __future__ import division

"""
This file exist for backward compatibility and presets various 
classes under their old name:

    STBPSTack = LayerStack

    FactoizedBernoulliTop, SigmoidBeliefLayer = SBNTop, SBN
    NADE, CNADE = NADETop, NADE

"""


from learning.models.rws import *

from learning.models.sbn import SBN, SBNTop
from learning.models.darn import DARN, DARNTop
from learning.models.nade import NADE, NADETop

STBPStack = LayerStack

SigmoidBeliefLayer, FactoizedBernoulliTop = SBN, SBNTop
CNADE, NADE = NADE, NADETop

