import re

a = """This PEP contains the index of all Python Enhancement Proposals,
known as PEPs.  PEP numbers are assigned by the PEP editors, and
once assigned are never changed[1].  The Mercurial history[2] of
the PEP texts represent their historical record.
"""

m = re.search(r'(\S+)\[(\d+)\]', a)
print m.group(0), m.group(1), m.group(2)

for m in re.finditer(r'(\S+)\[(\d+)\]', a):
    print m.group(0), m.group(1), m.group(2)

match_all = re.findall(r'(\S+)\[(\d+)\]',a)
print match_all

