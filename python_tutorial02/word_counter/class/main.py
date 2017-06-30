from fruit import Fruit, Banana, Orange

a = Fruit('apple','sweet')
print 'Initial price of the fruit', a.price
a.price = 10
print 'New price of the fruit', a.price

del a
print 'delete counter', Fruit.del_counter

b = Banana('monkey bannana','creamy')
b.squeeze()

del b
print 'delete counter', Fruit.del_counter

o = Orange('naval orange','yummy')
o.squeeze()

del o
print 'delete counter', Fruit.del_counter
