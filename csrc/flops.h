
#ifndef FLOPS_H
#define FLOPS_H

#ifdef TRACEFLOPS
#include <stdio.h>
#include <stdint.h>

#ifndef FLOPPATH
#define FLOPPATH "flopslog.csv"
#endif

static uint64_t total_flops = 0; //static so is is only consideder in the compilation entity

#define FLOP(n) total_flops += (uint64_t)(n)

static inline void flops_readout(int B) {
    FILE *fp = fopen(FLOPPATH, "a");

    if (fp != NULL) {
        fprintf(fp, "%d,%lu\n",B , total_flops);
        fclose(fp);
    } else {
        fprintf(stderr, "Error: Could not open file '%s' for writing flops.\n", FLOPPATH);
    }
}

#else

#define FLOP(n)
#define flops_readout(B)

#endif

#endif //FLOPS_H
