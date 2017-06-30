import re

a = """This PEP contains the index of all Python Enhancement Proposals,
known as PEPs.  PEP numbers are assigned by the PEP editors, and
once assigned are never changed[1].  The Mercurial history[2] of
the PEP texts represent their historical record.
"""

m = re.sub(r'\[(\d+)\]','{\g<1>}', a)
print a
print m
