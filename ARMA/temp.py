# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:52:55 2018

@author: Daan
"""

found_attacks = set()
found_attacks.add(1)
found_attacks.add(2)
found_attacks.add(3)

other_attacks = set()
other_attacks.add(4)
other_attacks.add(5)
print(found_attacks)
print(other_attacks)

found_attacks.update(other_attacks)
print(found_attacks)