import os
import shutil
import re

FILE_PATH = '/Users/danieltemkin/Repos/TriDither_AlternativePixelHistory/out/'
LOG_PATH = '/Users/danieltemkin/Repos/TriDither_AlternativePixelHistory/logs/'
COPY_TO = '/Users/danieltemkin/Repos/TriDither_AlternativePixelHistory/out/compiled/'

loc = '2022_8_3_22_35_35_579427'

"""Reads log, copies all the files in order they were generated, into a single folder where they can be loaded as frames in Photoshop
"""

if __name__ == '__main__':
    file1 = open(f'{LOG_PATH}exh_{loc}.log', 'r')
    lines = file1.readlines()

    guidlist = dict()
    currcell = []
    
    # Strips the newline character
    for line in lines:
        line = line.strip()

        gennum = re.search(r'Beginning generation ([0-9]+)', line)
        if gennum is not None:
            currgen = int(gennum.group(1))        
        cellnum = re.search(r'Getting score for #([0-9]+)', line)
        if cellnum is not None:
            currcell.append(int(cellnum.group(1)))
        cap = re.search(r'BEGIN [a-zA-Z ]+ TEST FOR ([0-9a-z-]+)', line)
        if cap is not None:
            guidlist[currcell.pop(0)] = [currgen, cap.group(1)]

    for i in guidlist.keys():
        print(guidlist[i])
        shutil.copy(f'{FILE_PATH}{loc}/{guidlist[i][0]}/{guidlist[i][1]}/hexagon.png', f'{COPY_TO}hex/hex_{"{:04d}".format(i)}.png')

