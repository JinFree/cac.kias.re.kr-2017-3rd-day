import argparse

parser = argparse.ArgumentParser(description='argument process.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='an integer for the accumulator')
parser.add_argument('-a', dest='a_option', type=int, help='a option')
parser.add_argument('-b', dest='b_option', help='b option')
parser.add_argument('-c', dest='c_option', action='store_true', help='c option')

args = parser.parse_args()

print args.integers
print 'a:', args.a_option 
print 'b:', args.b_option 
print 'c:', args.c_option 
