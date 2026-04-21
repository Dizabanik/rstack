#include "rstack.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Compiler hints for likely/unlikely statements (helps with branch prediction).
 */
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

/* Defines alignment of rstack struct. */
#define RSTACK_ALIGNMENT 64

/* Fixed overhead of rstack (excluding sao_data array). */
#define RSTACK_FIXED_OVERHEAD                                                  \
	(2 * sizeof(size_t) + 4 * sizeof(void *) + 2 * sizeof(uint32_t) +          \
	 sizeof(uint64_t))

/* This preproc definition checks whether N is a valid size of sao_data array
 * for rstack to be multiple of RSTACK_ALIGNMENT. */
#define SAO_OK(N)                                                              \
	(((RSTACK_FIXED_OVERHEAD + (N) * sizeof(uint64_t)) % RSTACK_ALIGNMENT) == 0)

/* This definition finds the smallest N >= 4 and N <=  24 such that
 * SAO_OK(N) is true. (in truth, we expect it to be 8, but we also try to keep
 * it fast on some 'strange' systems or different compilares from gcc and
 * clang.)
 */
#define SAO_CAP_FIND                                                           \
	(SAO_OK(4)	  ? 4                                                          \
	 : SAO_OK(5)  ? 5                                                          \
	 : SAO_OK(6)  ? 6                                                          \
	 : SAO_OK(7)  ? 7                                                          \
	 : SAO_OK(8)  ? 8                                                          \
	 : SAO_OK(9)  ? 9                                                          \
	 : SAO_OK(10) ? 10                                                         \
	 : SAO_OK(11) ? 11                                                         \
	 : SAO_OK(12) ? 12                                                         \
	 : SAO_OK(13) ? 13                                                         \
	 : SAO_OK(14) ? 14                                                         \
	 : SAO_OK(15) ? 15                                                         \
	 : SAO_OK(16) ? 16                                                         \
	 : SAO_OK(17) ? 17                                                         \
	 : SAO_OK(18) ? 18                                                         \
	 : SAO_OK(19) ? 19                                                         \
	 : SAO_OK(20) ? 20                                                         \
	 : SAO_OK(21) ? 21                                                         \
	 : SAO_OK(22) ? 22                                                         \
	 : SAO_OK(23) ? 23                                                         \
	 : SAO_OK(24) ? 24                                                         \
				  : -1)

#define SAO_CAP SAO_CAP_FIND

/* We assert using preprocessor to ensure we have found right size of SAO array
 * AND that pointer fits in uint64_t. */
_Static_assert(SAO_CAP >= 4,
			   "No valid SAO capacity >= 4 found in search range");
_Static_assert(sizeof(uintptr_t) <= sizeof(uint64_t),
			   "Pointer must fit in a uint64_t element slot");

/* Number of uint64_t needed for bitset to hold `cap` bits. To find that, we
 * need to divide by 64 and round up. So, (cap + 63) / 64. Division by 64 is
 * identical to bitshift by 6 (and bitshifting is the fastest way for CPU to do
 * division), thus this formula.*/
#define TYPE_WORDS(cap) (((cap) + 63) >> 6)

/* Size of each arena block (64 KB). */
#define ARENA_BLOCK_SIZE 65536
/* Actual stacks start at offset ARENA_HEADER_SIZE. */
#define ARENA_HEADER_SIZE 64

typedef struct arena_block {
	struct arena_block *next;
	size_t bump; /* index where next stack should be allocated in this block */
} arena_block_t;

_Static_assert(sizeof(arena_block_t) <= ARENA_HEADER_SIZE,
			   "Arena header must fit within ARENA_HEADER_SIZE");

/* The main rstack structure, which includes our GC layout. */
struct rstack {
	/* On x86-64 linux with GCC, we expect this to be 64 bytes. */
	size_t size;		 /* current element count */
	size_t capacity;	 /* if 0, it is SAO mode, if > 0, then it is dynamic
							mode */
	uint64_t *block;	 /* dynamic mode data */
	rstack_t *root_next; /* doubly-linked root set / free list */
	rstack_t *root_prev; /* doubly-linked root set */
	rstack_t *work_next; /* GC traversal worklist. Used because dynamically
							allocating an array for them is slow AND disallows
							us to try to free memory using GC when OOM */
	uint32_t gc_epoch;	 /* epoch stamp for GC marking*/
	uint32_t visit_gen;	 /* DFS cycle detection stamp */
	uint64_t sao_type;	 /* packed type bits for SAO mode */

	/* SAO inline data. On linux x86-64 with GCC, we also expect it to be 64
	 * bytes. */
	uint64_t sao_data[SAO_CAP]; /* SAO array of elements */
};

_Static_assert(sizeof(struct rstack) ==
				   RSTACK_FIXED_OVERHEAD + SAO_CAP * sizeof(uint64_t),
			   "Unexpected size of struct rstack");
_Static_assert(sizeof(struct rstack) % RSTACK_ALIGNMENT == 0,
			   "struct rstack size must be a multiple of RSTACK_ALIGNMENT");

#define RSTACK_SIZE ((size_t)sizeof(struct rstack))

/* Max amount of elements in one arena block. */
#define ARENA_BLOCK_CAP                                                        \
	((ARENA_BLOCK_SIZE - ARENA_HEADER_SIZE) / sizeof(rstack_t))

_Static_assert(ARENA_BLOCK_CAP >= 1, "Arena block too small for one stack");

/* DFS frame. */
typedef struct {
	rstack_t *stack;
	size_t index; /* next element to examine */
} dfs_frame_t;

#define DFS_LOCAL_CAP 2048 /* 32 KB of local DFS stack */

/* Arena system. Each block holds ARENA_BLOCK_CAP stacks, all 64-byte aligned.
 * Free list reuses root_next pointer in freed stacks, resulting in less
 * allocations. */

static arena_block_t *g_arena_blocks = nullptr;
static rstack_t *g_free_list = nullptr;
static size_t g_total_alive = 0;
static size_t g_alive_after_last_gc = 0;

/* Get pointer to first stack slot in a block. */
static inline rstack_t *block_stacks(arena_block_t *blk) {
	return (rstack_t *)((char *)blk + ARENA_HEADER_SIZE);
}

/* Free an arena block. */
static inline void arena_free_block(arena_block_t *blk) {
	/* The real malloc pointer is stored at ((void**)blk)[-1]. */
	void *raw = ((void **)blk)[-1];
	free(raw);
}

/* Allocate an aligned arena block using over-allocating malloc. Basically a
 * custom replacement for aligned_alloc (or posix_memalign). I have asked on
 * forum, but got an answer that normal aligned_alloc will not pass the tests.
 */
static arena_block_t *arena_alloc_block(void) {
	size_t total = ARENA_BLOCK_SIZE + (RSTACK_ALIGNMENT - 1) + sizeof(void *);
	void *raw = malloc(total);
	if (unlikely(!raw))
		return nullptr;

	uintptr_t addr = (uintptr_t)raw + sizeof(void *) + (RSTACK_ALIGNMENT - 1);
	/* It works like that: we make the alignment to have 0s in the end. Using
	 * bitwise AND, we ensure that our adress is a multiple of RSTACK_ALIGNMENT.
	 */
	addr &= ~(uintptr_t)(RSTACK_ALIGNMENT - 1);
	((void **)addr)[-1] = raw; /* stash real pointer for free() */

	arena_block_t *blk = (arena_block_t *)addr;
	blk->next = nullptr;
	blk->bump = 0;
	return blk;
}

/* Free all arena blocks (called when g_total_alive reaches 0). */
static void arena_cleanup(void) {
	g_free_list = nullptr;
	while (g_arena_blocks) {
		arena_block_t *next = g_arena_blocks->next;
		arena_free_block(g_arena_blocks);
		g_arena_blocks = next;
	}
}

static void gc_collect(void);

/* Allocate an rstack_t from the arena.
 * Triggers GC when arena is exhausted to reclaim unreachable stacks. */
static rstack_t *alloc_stack(void) {
	rstack_t *rs;
	/* Here, we will use offsetof(rstack_t, sao_data) for memset, as we anyway
	 * don't use any SAO data until certain slot is being filled, thus saving
	 * some CPU cycles.*/

	/* Try the free list */
	if (g_free_list) {
		rs = g_free_list;
		g_free_list = rs->root_next;
		memset(rs, 0, offsetof(rstack_t, sao_data));
		return rs;
	}

	/* Try allocating from the newest block */
	if (g_arena_blocks && g_arena_blocks->bump < ARENA_BLOCK_CAP) {
		rs = &block_stacks(g_arena_blocks)[g_arena_blocks->bump++];
		memset(rs, 0, offsetof(rstack_t, sao_data));
		return rs;
	}

	/* Arena exhausted. We trigger GC only if the heap has grown significantly
	 * since the last collection (to avoid doing it when we just add-remove
	 * element each time, for example). */
	if (g_total_alive > g_alive_after_last_gc + (g_alive_after_last_gc / 2) ||
		g_alive_after_last_gc == 0) {
		gc_collect();

		if (g_free_list) {
			rs = g_free_list;
			g_free_list = rs->root_next;
			memset(rs, 0, offsetof(rstack_t, sao_data));
			return rs;
		}
	}

	/* Still no memory, so we should allocate a new arena block. */
	arena_block_t *blk = arena_alloc_block();
	if (unlikely(!blk))
		return nullptr;
	blk->next = g_arena_blocks;
	g_arena_blocks = blk;
	rs = &block_stacks(blk)[blk->bump++];
	memset(rs, 0, offsetof(rstack_t, sao_data));
	return rs;
}

/* Garbage Collector. It is based on root set, which is a doubly-linked list of
 * stacks that the user still has access to (haven't been rstack_delete'd). */

static rstack_t *g_roots_head = nullptr; /* Doubly linked list of 'roots' */
static size_t g_root_count = 0;
static uint32_t g_current_epoch = 1; /* global GC epoch (0 = uninit/free) */
static uint32_t g_visit_gen = 0;	 /* DFS visit generation counter */

/* Add root (rstack) to linked list. */
static inline void root_add(rstack_t *rs) {
	rs->root_next = g_roots_head;
	rs->root_prev = nullptr;
	if (g_roots_head)
		g_roots_head->root_prev = rs;
	g_roots_head = rs;
	g_root_count++;
}

/* Remove root (rstack) from linked list. */
static inline void root_remove(rstack_t *rs) {
	if (rs->root_prev)
		rs->root_prev->root_next = rs->root_next;
	else
		g_roots_head = rs->root_next;
	if (rs->root_next)
		rs->root_next->root_prev = rs->root_prev;
	rs->root_next = nullptr;
	rs->root_prev = nullptr;
	g_root_count--;
}

/* Check if rs is currently in the root set. */
static inline bool is_root(const rstack_t *rs) {
	return rs == g_roots_head || rs->root_prev != nullptr;
}

/* Advance visit_get. Skip 0 on wrap. */
static inline uint32_t next_visit_gen(void) {
	if (unlikely(++g_visit_gen == 0)) {
		g_visit_gen = 1;
		/* Global reset to prevent collision. */
		for (arena_block_t *blk = g_arena_blocks; blk; blk = blk->next) {
			rstack_t *slots = block_stacks(blk);
			for (size_t i = 0; i < blk->bump; i++)
				slots[i].visit_gen = 0;
		}
	}
	return g_visit_gen;
}

/* Pointer to element data array (SAO or dynamic). */
static inline uint64_t *rs_data(const rstack_t *rs) {
	if (rs->capacity == 0)
		return (uint64_t *)rs->sao_data;
	return rs->block + TYPE_WORDS(rs->capacity);
}

/* Is element i a sub-stack reference? Returns 0 if raw value, and 1 if
 * sub-stack pointer. */
static inline bool rs_type_at(const rstack_t *rs, size_t i) {
	if (rs->capacity == 0)
		/* Shifts the bit to the right and checks if it's 1*/
		return (rs->sao_type >> i) & 1;
	/* i >> 6 <=> i / 64. i & 63 <=> i % 64. Shifts the bit to the right and
	 * checks if it's 1*/
	return (rs->block[i >> 6] >> (i & 63)) & 1;
}
/* Sets type bit for element i for SAO mode. */
#define RS_SET_TYPE_SAO(rs, i, is_stack)                                       \
	if (is_stack)                                                              \
		rs->sao_type |=                                                        \
			(1ULL << i); /* Bitwise OR is enough for is_stack = true */        \
	else                                                                       \
		rs->sao_type &=                                                        \
			~(1ULL << i); /* NOT makes it so we have 0 ONLY on position i */

/* Sets type bit for element i for dynamic mode. */
#define RS_SET_TYPE_DYNAMIC(rs, i, is_stack)                                   \
	if (is_stack)                                                              \
		rs->block[i >> 6] |= (1ULL << (i & 63));                               \
	else                                                                       \
		rs->block[i >> 6] &= ~(1ULL << (i & 63));

/* Set type bit for element i. */
__attribute__((always_inline)) static inline void
rs_set_type(rstack_t *rs, size_t i, bool is_stack) {
	if (rs->capacity == 0) {
		RS_SET_TYPE_SAO(rs, i, is_stack);
	} else {
		RS_SET_TYPE_DYNAMIC(rs, i, is_stack);
	}
}

/* Read element i as a sub-stack pointer. */
static inline rstack_t *rs_get_stack(const rstack_t *rs, size_t i) {
	return (rstack_t *)(uintptr_t)rs_data(rs)[i];
}

/* Capacity manager. Returns 0 on success, -1 on ENOMEM. After success, at least
 * one more element can be pushed. */
__attribute__((noinline)) static int ensure_capacity(rstack_t *rs) {
	if (rs->capacity == 0) {
		/* SAO mode - still room? */
		if (likely(rs->size < (size_t)SAO_CAP))
			return 0;

		/* Transition from SAO to dynamic. */
		size_t new_cap = (size_t)SAO_CAP * 2;
		size_t tw = TYPE_WORDS(new_cap);
		uint64_t *new_block = malloc((tw + new_cap) * sizeof(uint64_t));
		if (unlikely(!new_block)) {
			/* Try asking GC to collect data, as it may free some space. */
			gc_collect();
			new_block = malloc((tw + new_cap) * sizeof(uint64_t));
			if (unlikely(!new_block)) {
				/* Truly out of memory. */
				errno = ENOMEM;
				return -1;
			}
		}

		/* Fill with 0, then copy existing bits from sao_type. */
		memset(new_block, 0, tw * sizeof(uint64_t));
		new_block[0] = rs->sao_type;

		/* Copy existing data. */
		memcpy(new_block + tw, rs->sao_data, rs->size * sizeof(uint64_t));

		rs->block = new_block;
		rs->capacity = new_cap;
		return 0;
	}

	/* Dynamic mode - still room? */
	if (likely(rs->size < rs->capacity))
		return 0;

	/* Grow 2x. */
	size_t new_cap = rs->capacity * 2;
	size_t old_tw = TYPE_WORDS(rs->capacity);
	size_t new_tw = TYPE_WORDS(new_cap);

	if (old_tw == new_tw) {
		/* Type-words count unchanged, so we realloc in-place. */
		uint64_t *nb =
			realloc(rs->block, (new_tw + new_cap) * sizeof(uint64_t));
		if (unlikely(!nb)) {
			/* Try asking GC to collect data, as it may free some space. */
			gc_collect();
			nb = realloc(rs->block, (new_tw + new_cap) * sizeof(uint64_t));
			if (unlikely(!nb)) {
				/* Truly out of memory. */
				errno = ENOMEM;
				return -1;
			}
		}
		rs->block = nb;
		rs->capacity = new_cap;
	} else {
		/* Type-words count changed, so the whole data must shift. */
		uint64_t *nb = malloc((new_tw + new_cap) * sizeof(uint64_t));
		if (unlikely(!nb)) {
			/* Try asking GC to collect data, as it may free some space. */
			gc_collect();
			nb = malloc((new_tw + new_cap) * sizeof(uint64_t));
			if (unlikely(!nb)) {
				/* Truly out of memory. */
				errno = ENOMEM;
				return -1;
			}
		}
		/* Copy old type words. */
		memcpy(nb, rs->block, old_tw * sizeof(uint64_t));
		/* Zero new type words. */
		memset(nb + old_tw, 0, (new_tw - old_tw) * sizeof(uint64_t));
		/* Copy data from old offset to new offset. */
		memcpy(nb + new_tw, rs->block + old_tw, rs->size * sizeof(uint64_t));
		free(rs->block);
		rs->block = nb;
		rs->capacity = new_cap;
	}
	return 0;
}

/* These are macros for GC traversal, helping us to traverse children. */

/* SAO mode. */
#define SAO_CHILDREN(cur, BODY)                                                \
	do {                                                                       \
		const uint64_t *_d = rs_data(cur);                                     \
		size_t _sz = (cur)->size;                                              \
		uint64_t _m = (cur)->sao_type & ((1ULL << _sz) - 1);                   \
		while (_m) {                                                           \
			size_t idx = (size_t)__builtin_ctzll(_m);                          \
			_m &= _m - 1;                                                      \
			rstack_t *child = (rstack_t *)(uintptr_t)_d[idx];                  \
			{                                                                  \
				BODY                                                           \
			}                                                                  \
		}                                                                      \
	} while (0)

/* Dynamic mode. */
#define DYN_CHILDREN(cur, BODY)                                                \
	do {                                                                       \
		const uint64_t *_d = rs_data(cur);                                     \
		size_t _sz = (cur)->size;                                              \
		if (_sz == 0)                                                          \
			break;                                                             \
		size_t _active_tw = (_sz + 63) >> 6;                                   \
		for (size_t _w = 0; _w < _active_tw; _w++) {                           \
			uint64_t _b = (cur)->block[_w];                                    \
			if (_w == _active_tw - 1 && (_sz & 63))                            \
				_b &= (1ULL << (_sz & 63)) - 1;                                \
			while (_b) {                                                       \
				size_t _bit = (size_t)__builtin_ctzll(_b);                     \
				_b &= _b - 1;                                                  \
				size_t idx = _w * 64 + _bit;                                   \
				rstack_t *child = (rstack_t *)(uintptr_t)_d[idx];              \
				{                                                              \
					BODY                                                       \
				}                                                              \
			}                                                                  \
		}                                                                      \
	} while (0)

/* Iterate through each child. */
#define EACH_CHILD(cur, BODY)                                                  \
	if ((cur)->capacity == 0) {                                                \
		SAO_CHILDREN(cur, BODY);                                               \
	} else {                                                                   \
		DYN_CHILDREN(cur, BODY);                                               \
	}
/* Epoch-based M&S GC. O(1) initialization using epoch comparison.
 * 1. Advance the global epoch.
 * 2. Go from the roots. We mark all reachable stacks with the new epoch.
 * 3. Sweep every arena slot.
 * 4. If nothing is alive, release all arena memory. */

static void gc_collect(void) {
	/* Advance epoch (skip 0, as it is reserved for free/uninit). */
	g_current_epoch++;
	if (unlikely(g_current_epoch == 0)) {
		/* Full reset, we walk the arena and set every active epoch to 1. */
		g_current_epoch = 2;
		for (arena_block_t *blk = g_arena_blocks; blk; blk = blk->next) {
			rstack_t *slots = block_stacks(blk);
			for (size_t i = 0; i < blk->bump; i++) {
				slots[i].gc_epoch =
					1; /* Reset all live memory to safe baseline */
			}
		}
	}

	uint32_t epoch = g_current_epoch;

	/* Go from roots: O(N+E) */
	rstack_t *work = nullptr;

	for (rstack_t *r = g_roots_head; r; r = r->root_next) {
		r->gc_epoch = epoch;
		r->work_next = work;
		work = r;
	}

	while (work) {
		rstack_t *cur = work;
		work = cur->work_next;

		EACH_CHILD(cur, {
			if (child->gc_epoch != epoch) {
				child->gc_epoch = epoch;
				child->work_next = work;
				work = child;
			}
		});
	}

	/* Arena sweep and free list rebuild. */
	g_free_list = nullptr;
	g_total_alive = 0;

	for (arena_block_t *blk = g_arena_blocks; blk; blk = blk->next) {
		rstack_t *slots = block_stacks(blk);
		for (size_t i = 0; i < blk->bump; i++) {
			rstack_t *s = &slots[i];
			if (s->gc_epoch != epoch) {
				/* Garbage. So, we free dynamic block and return it to free
				 * list.
				 */
				if (s->capacity > 0) {
					free(s->block);
					s->capacity = 0;
					s->block = nullptr;
				}
				s->root_next = g_free_list;
				g_free_list = s;
			} else {
				g_total_alive++;
			}
		}
	}

	/* If everything is dead, release all arena memory. */
	if (g_total_alive == 0)
		arena_cleanup();

	g_alive_after_last_gc = g_total_alive; /* Remember heap size */
}

/* Push a frame, growing heap buffer if needed.
 * Returns 0 on success, -1 on ENOMEM. */
static inline int dfs_push(dfs_frame_t **frames, size_t *cap, size_t *top,
						   dfs_frame_t *local, rstack_t *stack, size_t index) {
	if (*top >= *cap) {
		size_t new_cap = *cap * 2;
		dfs_frame_t *nf;
		if (*frames == local) {
			nf = malloc(new_cap * sizeof(dfs_frame_t));
			if (unlikely(!nf))
				return -1;
			memcpy(nf, local, *top * sizeof(dfs_frame_t));
		} else {
			nf = realloc(*frames, new_cap * sizeof(dfs_frame_t));
			if (unlikely(!nf))
				return -1;
		}
		*frames = nf;
		*cap = new_cap;
	}
	(*frames)[*top] = (dfs_frame_t){stack, index};
	(*top)++;
	return 0;
}

static inline void dfs_cleanup(dfs_frame_t *frames, dfs_frame_t *local) {
	if (frames != local)
		free(frames);
}

rstack_t *rstack_new(void) {
	rstack_t *rs = alloc_stack();
	if (unlikely(!rs)) {
		errno = ENOMEM;
		return nullptr;
	}
	/* alloc_stack already zeroed the struct. */
	rs->gc_epoch = g_current_epoch;
	g_total_alive++;
	root_add(rs);
	return rs;
}

void rstack_delete(rstack_t *rs) {
	if (!rs)
		return;

	/* Check to not double-delete. */
	if (!is_root(rs))
		return;

	root_remove(rs);

	/* When all user handles are gone, everything unreachable is garbage. */
	if (g_root_count == 0)
		gc_collect();
}

/* Cold path. Only called when memory allocation is required. */
__attribute__((noinline)) static int push_cold(rstack_t *rs, uint64_t v,
											   bool is_stack) {
	if (unlikely(ensure_capacity(rs) != 0))
		return -1;

	rs_data(rs)[rs->size] = v;
	rs_set_type(rs, rs->size, is_stack);
	rs->size++;
	return 0;
}

/* Hot path. Moves CPU-intensive capacity checks as far as possible. */
__attribute__((always_inline)) static inline int
push_hot(rstack_t *rs, uint64_t v, bool is_stack) {
	size_t sz = rs->size;
	if (rs->capacity == 0) {
		if (likely(sz < (size_t)SAO_CAP)) {
			rs->sao_data[sz] = v;
			RS_SET_TYPE_SAO(rs, sz, is_stack);
			rs->size = sz + 1;
			return 0;
		}
	} else {
		if (likely(sz < rs->capacity)) {
			size_t tw = TYPE_WORDS(rs->capacity);
			rs->block[tw + sz] = v;
			RS_SET_TYPE_DYNAMIC(rs, sz, is_stack);
			rs->size = sz + 1;
			return 0;
		}
	}

	/* Fallback to allocation. */
	return push_cold(rs, v, is_stack);
}

int rstack_push_value(rstack_t *rs, uint64_t value) {
	if (unlikely(!rs)) {
		errno = EINVAL;
		return -1;
	}
	return push_hot(rs, value, false);
}

int rstack_push_rstack(rstack_t *rs1, rstack_t *rs2) {
	if (unlikely(!rs1 || !rs2)) {
		errno = EINVAL;
		return -1;
	}
	return push_hot(rs1, (uint64_t)(uintptr_t)rs2, true);
}

void rstack_pop(rstack_t *rs) {
	if (!rs || rs->size == 0)
		return;
	rs->size--;
}

/* "Recursively" (iteratively via DFS) checks whether the stack contains any
 * value. If all elements are sub-stacks, returns true.
 */

bool rstack_empty(rstack_t *rs) {
	if (!rs || rs->size == 0)
		return true;

	uint32_t gen = next_visit_gen();

	/* Quick check for SAO mode. Does it contain ANY value (bit = 0)? */
	if (rs->capacity == 0) {
		uint64_t size_mask = (1ULL << rs->size) - 1;
		/* If ~sao_type AND size_mask has any bits set, we have a value. */
		if (~rs->sao_type & size_mask)
			return false;
	}

	rs->visit_gen = gen;

	dfs_frame_t local[DFS_LOCAL_CAP];
	dfs_frame_t *frames = local;
	size_t frcap = DFS_LOCAL_CAP;
	size_t frtop = 0;

	if (dfs_push(&frames, &frcap, &frtop, local, rs, 0) != 0)
		return true; /* OOM, we treat as empty. */

	bool result = true;

	while (frtop > 0) {
		dfs_frame_t *top = &frames[frtop - 1];
		rstack_t *cur = top->stack;

		if (top->index >= cur->size) {
			/* Finished traversing this stack, so we pop it. */
			frtop--;
			continue;
		}

		size_t i = top->index++;

		if (!rs_type_at(cur, i)) {
			/* We found a value. */
			result = false;
			break;
		}

		/* It's a sub-stack. Descend if not visited. */
		rstack_t *child = rs_get_stack(cur, i);
		if (child->visit_gen == gen || child->size == 0)
			continue;

		child->visit_gen = gen;
		__builtin_prefetch(rs_data(child), 0, 1);

		/* Quick scan of child's types. */
		if (child->capacity == 0) {
			uint64_t child_mask = (1ULL << child->size) - 1;
			if (~child->sao_type & child_mask) {
				result = false;
				break;
			}
		}

		if (top->index == cur->size) {
			/* This was the last element of 'cur'. Overwrite the current
			 * frame.*/
			top->stack = child;
			top->index = 0; /* empty iterates forwards, so start child at 0 */
			continue;
		}

		/* Descend into the child stack. */
		if (dfs_push(&frames, &frcap, &frtop, local, child, 0) != 0) {
			result = true; /* OOM, so return empty. */
			break;
		}
	}

	dfs_cleanup(frames, local);
	return result;
}

/* Finds the value closest to the top of the (recursive) stack.
 * Iterative DFS, visiting sub-stacks from top element downward.
 * Visit generation prevents revisiting the same sub-stack.
 */

result_t rstack_front(rstack_t *rs) {
	result_t none = {false, 0};

	if (!rs || rs->size == 0)
		return none;

	uint32_t gen = next_visit_gen();
	rs->visit_gen = gen;

	dfs_frame_t local[DFS_LOCAL_CAP];
	dfs_frame_t *frames = local;
	size_t frcap = DFS_LOCAL_CAP;
	size_t frtop = 0;

	/* Start from top.  */
	if (dfs_push(&frames, &frcap, &frtop, local, rs, rs->size) != 0)
		return none;

	while (frtop > 0) {
		dfs_frame_t *top = &frames[frtop - 1];
		rstack_t *cur = top->stack;

		if (top->index == 0) {
			/* Finished traversing this stack, so we pop it. */
			frtop--;
			continue;
		}

		size_t i = --top->index; /* iterate top-to-bottom */

		if (!rs_type_at(cur, i)) {
			/* We found a value. */
			result_t res = {true, rs_data(cur)[i]};
			dfs_cleanup(frames, local);
			return res;
		}

		/* It's a sub-stack, so descend if not visited. */
		rstack_t *child = rs_get_stack(cur, i);
		if (child->visit_gen == gen || child->size == 0)
			continue;

		child->visit_gen = gen;
		__builtin_prefetch(rs_data(child), 0, 1);

		if (top->index == 0) {
			/* This was the last element of 'cur'. Overwrite the current frame.
			 */
			top->stack = child;
			top->index = child->size;
			continue;
		}

		/* Descend into the child stack. */
		if (dfs_push(&frames, &frcap, &frtop, local, child, child->size) != 0) {
			dfs_cleanup(frames, local);
			return none;
		}
	}

	dfs_cleanup(frames, local);
	return none;
}

/* Two-pass file parser:
 *   1. validate content (only digits + whitespace) and count numbers.
 *   2. parse numbers, push values.
 * Pre-allocates capacity after pass 1 for a single data allocation.
 */

rstack_t *rstack_read(char const *path) {
	if (unlikely(!path)) {
		errno = EINVAL;
		return nullptr;
	}

	FILE *f = fopen(path, "r");
	if (unlikely(!f))
		return nullptr;

	/* Determine file size using fseek/ftell. */
	if (fseek(f, 0, SEEK_END) != 0) {
		fclose(f);
		return nullptr;
	}
	long fsize = ftell(f);
	if (fsize < 0) {
		fclose(f);
		return nullptr;
	}
	if (fseek(f, 0, SEEK_SET) != 0) {
		fclose(f);
		return nullptr;
	}

	/* Handle empty file. */
	if (fsize == 0) {
		fclose(f);
		return rstack_new();
	}

	/* Read entire file into buffer. */
	char *buf = malloc((size_t)fsize + 1);
	if (unlikely(!buf)) {
		/* Try asking GC to collect data, as it may free some space. */
		gc_collect();
		buf = malloc((size_t)fsize + 1);
		if (unlikely(!buf)) {
			/* Truly out of memory. */
			fclose(f);
			errno = ENOMEM;
			return nullptr;
		}
	}

	size_t nread = fread(buf, 1, (size_t)fsize, f);
	fclose(f);

	if (unlikely(nread != (size_t)fsize)) {
		free(buf);
		errno = EIO;
		return nullptr;
	}
	buf[nread] = '\0';

	/* Pass 1. */
	size_t count = 0;
	const char *p = buf;
	while (*p) {
		/* Skip whitespace. */
		while (*p && isspace((unsigned char)*p))
			p++;
		if (!*p)
			break;

		/* Must be a digit (no +/- signs). */
		if (!isdigit((unsigned char)*p)) {
			free(buf);
			errno = EINVAL;
			return nullptr;
		}

		/* Skip the number. */
		while (*p && isdigit((unsigned char)*p))
			p++;

		/* After number, it must be whitespace or EOF. */
		if (*p && !isspace((unsigned char)*p)) {
			free(buf);
			errno = EINVAL;
			return nullptr;
		}
		count++;
	}

	/* Create stack. */
	rstack_t *rs = rstack_new();
	if (unlikely(!rs)) {
		free(buf);
		return nullptr;
	}

	/* Pre-allocate if needed (avoids repeated growth). */
	if (count > (size_t)SAO_CAP) {
		size_t cap = (size_t)SAO_CAP * 2;
		while (cap < count)
			cap *= 2;
		size_t tw = TYPE_WORDS(cap);
		uint64_t *blk = malloc((tw + cap) * sizeof(uint64_t));
		if (unlikely(!blk)) {
			/* Try asking GC to collect data, as it may free some space. */
			gc_collect();
			blk = malloc((tw + cap) * sizeof(uint64_t));
			if (unlikely(!blk)) {
				/* Truly out of memory. */
				free(buf);
				rstack_delete(rs);
				errno = ENOMEM;
				return nullptr;
			}
		}
		memset(blk, 0,
			   tw * sizeof(uint64_t)); /* all type bits = 0 (raw value) */
		rs->block = blk;
		rs->capacity = cap;
	}

	/* Pass 2. */
	p = buf;
	uint64_t *data = rs_data(rs);

	/* Constants for overflow checks later. */
	const uint64_t MAX_VAL_DIV_10 = UINT64_MAX / 10;
	const uint8_t MAX_VAL_MOD_10 = 5; /* UINT64_MAX % 10 */

	for (size_t n = 0; n < count; n++) {
		while (*p && isspace((unsigned char)*p))
			p++;

		uint64_t val = 0;
		const char *start = p;

		while (*p >= '0' && *p <= '9') {
			uint8_t digit = *p - '0';

			/* 2-cycle constant comparison instead of 40-cycle division */
			if (unlikely(val > MAX_VAL_DIV_10 ||
						 (val == MAX_VAL_DIV_10 && digit > MAX_VAL_MOD_10))) {
				free(buf);
				rstack_delete(rs);
				errno = ERANGE;
				return nullptr;
			}

			val = val * 10 + digit;
			p++;
		}

		/* Should never trigger if Pass 1 is correct. But, just to be safe. */
		if (unlikely(p == start)) {
			free(buf);
			rstack_delete(rs);
			errno = EINVAL;
			return nullptr;
		}

		data[n] = val;
	}
	rs->size = count;

	free(buf);
	return rs;
}

/* Lookup table for processing 2 digits at a time. It is suuuper cool and speeds
 * up our custom u64 to string conversion. */
static const char digit_pairs[201] = "0001020304050607080910111213141516171819"
									 "2021222324252627282930313233343536373839"
									 "4041424344454647484950515253545556575859"
									 "6061626364656667686970717273747576777879"
									 "8081828384858687888990919293949596979899";

/* We need it to optimize our rstack_write by replacing fprintf, which is
 * slooow */
static inline size_t fast_u64_to_str(uint64_t val, char *buf) {
	char temp[24];
	char *p = temp + 24;
	*(--p) = '\n';

	if (val == 0) {
		*(--p) = '0';
	} else {
		/* Process two digits at a time */
		while (val >= 100) {
			unsigned int const rem = val % 100;
			val /= 100;
			*(--p) = digit_pairs[rem * 2 + 1];
			*(--p) = digit_pairs[rem * 2];
		}
		/* Handle the remaining 1 or 2 digits */
		if (val < 10) {
			*(--p) = '0' + (char)val;
		} else {
			*(--p) = digit_pairs[val * 2 + 1];
			*(--p) = digit_pairs[val * 2];
		}
	}

	size_t len = (temp + 24) - p;
	/* gcc should optimize small, constant-bounded memcpy (we should hope it
	 * does). Although, even if not, memcpy is still fast due to vectorization.
	 */
	memcpy(buf, p, len);
	return len;
}

/* Iterative DFS. Uses visit_gen.
 * - Mark on DFS push.
 * - Clear on DFS pop.
 * - If child is already marked, then we have a cycle, so stop and return 0.
 * Returns 0 on success (including cycle-stopped), -1 on error (errno set).
 */

int rstack_write(char const *path, rstack_t *rs) {
	if (unlikely(!path || !rs)) {
		errno = EINVAL;
		return -1;
	}

	FILE *f = fopen(path, "w");
	if (unlikely(!f))
		return -1;

	/* Disable standard libc buffering to bypass the double-copy (because we
	 * have our own buffer). It should reduce memory usage and improve
	 * performance, although i am not sure about performance gain as buffer in
	 * libc is vectorized and optimized. */
	setvbuf(f, NULL, _IONBF, 0);

	uint32_t gen = next_visit_gen();
	rs->visit_gen = gen;

	dfs_frame_t local[DFS_LOCAL_CAP];
	dfs_frame_t *frames = local;
	size_t frcap = DFS_LOCAL_CAP;
	size_t frtop = 0;
	int ret = 0;
	bool cycle = false;

	/* 16KB Stack Buffer. Cache friendly, fast, no malloc failure risks. */
	char write_buf[16384];
	size_t buf_off = 0;

	if (dfs_push(&frames, &frcap, &frtop, local, rs, 0) != 0) {
		fclose(f);
		errno = ENOMEM;
		return -1;
	}

	while (frtop > 0 && !cycle) {
		dfs_frame_t *top = &frames[frtop - 1];
		rstack_t *cur = top->stack;

		if (top->index >= cur->size) {
			/* Finished traversing this stack, so we clear the visit generation
			 * to mark it as popped. */
			cur->visit_gen = 0;
			frtop--;
			continue;
		}

		size_t i = top->index++;

		if (!rs_type_at(cur, i)) {
			/* Flush buffer if near full (max number length is around 21 bytes,
			 * thus 32 is a safe margin). */
			if (unlikely(buf_off > sizeof(write_buf) - 32)) {
				if (fwrite(write_buf, 1, buf_off, f) != buf_off) {
					ret = -1;
					break;
				}
				buf_off = 0;
			}
			/* Write directly into the buffer. */
			buf_off += fast_u64_to_str(rs_data(cur)[i], write_buf + buf_off);
		} else {
			/* Sub-stack. */
			rstack_t *child = rs_get_stack(cur, i);

			if (child->visit_gen == gen) {
				/* Cycle detected, so we stop immediately. */
				cycle = true;
				break;
			}

			if (child->size == 0)
				continue; /* empty sub-stack, skip */

			child->visit_gen = gen; /* mark as on ancestor path */

			/* Descend into the child stack. */
			if (dfs_push(&frames, &frcap, &frtop, local, child, 0) != 0) {
				ret = -1;
				break;
			}
		}
	}

	/* Clean up ancestor marks for any stacks still on the DFS stack. */
	for (size_t i = 0; i < frtop; i++)
		frames[i].stack->visit_gen = 0;

	dfs_cleanup(frames, local);

	/* Final flush of any remaining data. */
	if (ret == 0 && buf_off > 0) {
		if (fwrite(write_buf, 1, buf_off, f) != buf_off)
			ret = -1;
	}

	if (ferror(f)) {
		int saved = errno;
		fclose(f);
		errno = saved;
		return -1;
	}

	fclose(f);
	return ret;
}
