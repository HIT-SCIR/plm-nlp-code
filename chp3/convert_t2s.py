# Defined in Section 3.4.3

import sys
import opencc

converter = opencc.OpenCC("t2s.json")
f_in = open(sys.argv[1], "r")

for line in f_in.readlines():
    line = line.strip()
    line_t2s = converter.convert(line)
    print(line_t2s)
