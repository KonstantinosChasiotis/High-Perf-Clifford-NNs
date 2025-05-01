
INCLUDE = -I../csrc

CFLAGS  = -O3 -DPMU -llapacke -lopenblas -DBUILD_MAIN $(INCLUDE)

SRCS    = bench_forward.c ../csrc/clifford_linear.c ../csrc/clifford_groupnorm.c

all: bench_main

bench_main: $(SRCS)
	gcc $(CFLAGS) $(SRCS) -o bench_main

clean:
	rm -f bench_main
