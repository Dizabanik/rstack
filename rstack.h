#ifndef _RSTACK_H
#define _RSTACK_H

#include <stddef.h>
#include <inttypes.h>

typedef struct rstack rstack_t;

typedef struct {
  bool     flag;  // To pole mówi, czy pole value zawiera wynik.
  uint64_t value; // W tym polu jest właściwy wynik.
} result_t;

rstack_t* rstack_new();
void      rstack_delete(rstack_t *rs);
int       rstack_push_value(rstack_t *rs, uint64_t value);
int       rstack_push_rstack(rstack_t *rs1, rstack_t *rs2);
void      rstack_pop(rstack_t *rs);
bool      rstack_empty(rstack_t *rs);
result_t  rstack_front(rstack_t *rs);
rstack_t* rstack_read(char const *path);
int       rstack_write(char const *path, rstack_t *rs);

#endif
