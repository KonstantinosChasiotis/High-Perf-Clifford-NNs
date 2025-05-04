
#ifndef FLOPS_H_
#define FLOPS_H_

#define TRACEFLOPS 0

unsigned long flops_add, flops_mult, flops_div;
#define FLOPS_RESET(folps) ((void) ((TRACEFLOPS) ? flops_add = flops_mult = flops_div = 0 : 0))
#define FLOP_ADD(folps) ((void) ((TRACEFLOPS) ? flops_add += folps : 0))
#define FLOP_MULT(folps) ((void) ((TRACEFLOPS) ? flops_mult += folps : 0))
#define FLOP_DIV(folps) ((void) ((TRACEFLOPS) ? flops_div += folps : 0))
#define FLOPS_READOUT() ((void) ((TRACEFLOPS) ? printf("FLOPS: \nADD , MULT , DIV \n%ld , %ld , %ld \n", flops_add, flops_mult, flops_div) : 0))

#endif //FLOPS_H_