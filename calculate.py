import os

def calculate_mean():
    f1 = open('tourist_result.txt', 'r')
    f2 = open('guide_result.txt', 'r')
    f3 = open('all_result.txt', 'r')

    s = 0
    for line in f1:
        s += float(line)
    print s * 0.2

    s = 0
    for line in f2:
        s += float(line)
    print s * 0.2

    s = 0
    for line in f3:
        s += float(line)
    print s * 0.2
    print '-' * 10


# traverse root directory, and list directories as dirs and files as files
d = list()
for root, dirs, files in os.walk("."):
    d = dirs
    break
d = sorted(d)[1:]
original = os.getcwd()
for dirs in d:
    if dirs == "result":
        continue
    os.chdir(dirs)
    print dirs
    calculate_mean()
    os.chdir(original)
