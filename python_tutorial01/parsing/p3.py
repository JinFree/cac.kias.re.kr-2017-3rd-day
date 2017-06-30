import re

a = """This PEP contains the index of all Python Enhancement Proposals,
known as PEPs.  PEP numbers are assigned by the PEP editors, and
once assigned are never changed[1].  The Mercurial history[2] of
the PEP texts represent their historical record.
"""

# single search
m = re.search(r'\[\d+\]', a)
print m.group(0)

# multiple search
for m in re.finditer(r'\[\d+\]',a):
    print m.group(0)

match_all = re.findall(r'\[\d+\]',a)
print match_all
