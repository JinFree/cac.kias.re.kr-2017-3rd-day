import Gnuplot
import numpy as np

x = range(10)
y = range(10)
z = [None]*10
for i in range(len(x)):
    z[i] = x[i]*y[i] 
plot_data = Gnuplot.Data(x,y,z)

p = Gnuplot.Gnuplot(debug=True)
p('set term X11')
p('set style data lines')
p('set title "An example of a 3D plot"')
p('set xlabel "x"')
p('set ylabel "y"')
p.splot(plot_data)
raw_input()
