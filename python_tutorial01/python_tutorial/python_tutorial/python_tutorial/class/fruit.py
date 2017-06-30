
class Fruit(object):
    del_counter = 0

    def __init__(self,name,taste):
        self.name = name
        self.taste = taste
        self._price = 0

    def squeeze(self):
        print 'You got a cup of', self.name, 'juice! It tastes', self.taste + '.'

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self,value):
        self._price = value

    def get_price(self):
        return self._price

    def set_price(self,value):
        self._price = value
        return self._price

    def set_name(self,value):
        self.name = value
        return value

    def get_name(self):
        return self.name

    def __del__(self):
        Fruit.del_counter += 1


class Banana(Fruit):
    def squeeze(self):
        print 'You got puree of', self.name + '! It tastes', self.taste + '.'


class Orange(Fruit):
    def squeeze(self):
        super(Orange,self).squeeze()
        print 'It also has pulp!'

