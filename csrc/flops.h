
#ifndef FLOPS_H
#define FLOPS_H

#ifdef TRACEFLOPS
#include <stdio.h>
#include <stdint.h>

static uint64_t total_flops = 0; //static so is is only consideder in the compilation entity

#define FLOP(n) total_flops += (uint64_t)(n)

static inline void flops_readout() {
    printf("FLOPS: %lu\n", total_flops);
}

#else

#define FLOP(n)
#define flops_readout()

#endif

#endif //FLOPS_H
