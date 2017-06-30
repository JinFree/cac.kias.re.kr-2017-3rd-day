PYCFLAGS := $(shell /usr/local/python-2.7/bin/python2.7-config --cflags)
PYLDFLAGS := $(shell /usr/local/python-2.7/bin/python2.7-config --ldflags)
CC = gcc
CFLAGS = -c -fpic $(PYCFLAGS)
LDFLAGS = $(PYLDFLAGS)
CYTHON = cython
OBJS = main.o pyend.o

.PRECIOUS: %.c

wrap.x: $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o wrap.x

%.c: %.pyx
	$(CYTHON) -v $<  -o $@

main.o: main.c pyend.h
	$(CC) $(CFLAGS) $<

pyend.o: pyend.c
	$(CC) $(CFLAGS) $<

pyend.h: pyend.c

clean:
	rm *.o *.x pyend.c pyend.h
