# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 14:06:10 2021

@author: Theodore B
"""

from heap import stack
from heap import plt_stack
import heap as h


stk = stack( (10, 10), (11, 21)  )  # xx, yy


  
plt_stack(stk)  


print('Spatial "grain" of heap: (dx, dy) =', h.space_grain(stk))


