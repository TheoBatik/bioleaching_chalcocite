# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:57:27 2021

@author: Theodore B
"""

import pint 


# Initialise the standard Pint Unit Registry
ureg = pint.UnitRegistry()

# Top tips with Pint:
# Check dimensionality with a.dimensionality
# Convert units using a.to method, e.g. speed.to('inch/minute')