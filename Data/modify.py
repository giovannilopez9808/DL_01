from os import listdir,system
from sys import argv
position=argv[1]
folder=argv[2]
data=argv[3]
files = listdir(folder)
files = sorted(files)
for i,file in enumerate(files):
    number = file.split("_")[1]
    filename = f"{folder}/{file}"
    filename2 = f"../../{data}/{position}_{folder}_{number}"
    # print(filename,filename2)
    command=f"mv {filename} {filename2}"
    system(command)

