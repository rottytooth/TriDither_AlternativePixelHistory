#import modules
from pathlib import Path
import shutil

#start_guid = '2022_10_9_17_46_21_850368'
start_guid = '2022_10_9_20_7_49_754854'
out_path = '../out_ps/me_rt_ex/'
i = 132

#define directory
dirpath=Path(f'../out/{start_guid}')

#search for files path with given name in dir and subdir using rglob
files=dirpath.rglob("hexagon_rev1.png")

#copy file in same directory with new name to rename
for path in files:
    copy_to = f'{out_path}me_rt{i}.png'
    print(f'{path} to {copy_to}')
    shutil.copy(path, copy_to, follow_symlinks=True)
    i += 1
    