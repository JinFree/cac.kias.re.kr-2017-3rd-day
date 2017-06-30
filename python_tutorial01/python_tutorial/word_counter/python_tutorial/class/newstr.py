
class Mystr(str):
    def strip(self):
        stripped = super(Mystr,self).strip()
        return '|' + stripped + '|'


a = Mystr('  abc  ')
b = a.strip()

print b
