#!/usr/bin/python3
# -*- coding: utf-8 -*-

import article6
import pandas as pd

#0 - no violation
#1 - violation
#3 - both
#2 - not ruled on
def admissibility_anal(desc):
    if 'No violation of Article 6' in desc:
        if 'Violation of Article 6' in desc:
            return 3
        else:
            return 0
    elif 'Violation of Article 3' in desc:
        return 1
    else:
        return 2


def admissibility_anal_simple(desc):
    if 'Violation of Article 3' in desc:
        return 1
    elif 'No violation of Article 6' in desc:
        return 0
    else:
        return 0

def main():
    pass

if __name__ == '__main__':
    main()
