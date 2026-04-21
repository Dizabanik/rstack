CC = gcc
CFLAGS = -Wall -Wextra -Wno-implicit-fallthrough -std=gnu23 -fPIC -O2
LDFLAGS = -shared -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=reallocarray -Wl,--wrap=free -Wl,--wrap=strdup -Wl,--wrap=strndup

SRCS = rstack.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean

all: librstack.so

librstack.so: $(OBJS) memory_tests.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

rstack.o: rstack.c rstack.h
memory_tests.o: memory_tests.c memory_tests.h

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f rstack.o memory_tests.o librstack.so