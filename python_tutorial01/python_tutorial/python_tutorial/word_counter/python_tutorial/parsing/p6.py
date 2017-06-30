import re

a = "a, b,  c, d,   e , f   , g"

print a.split(',')
print re.split(r'\s*,\s*',a)
