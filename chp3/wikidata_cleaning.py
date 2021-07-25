# Defined in Section 3.4.3

import sys
import re

def remove_empty_paired_punc(in_str):
    return in_str.replace('（）', '').replace('《》', '').replace('【】', '').replace('[]', '')
    
def remove_html_tags(in_str):
    html_pattern = re.compile(r'<[^>]+>', re.S)
    return html_pattern.sub('', in_str)

def remove_control_chars(in_str):
    control_chars = ''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))
    control_chars = re.compile('[%s]' % re.escape(control_chars))
    return control_chars.sub('', in_str)

f_in = open(sys.argv[1], 'r')
for line in f_in.readlines():
    line = line.strip()
    if re.search(r'^(<doc id)|(</doc>)', line):
        print(line)
        continue
    line = remove_empty_paired_punc(line)
    line = remove_html_tags(line)
    line = remove_control_chars(line)
    print(line)
