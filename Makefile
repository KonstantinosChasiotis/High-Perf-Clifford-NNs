# Makefile

# Compiler and flags
CC       := gcc
CFLAGS   := -O3 -fPIC -Wno-alloc-size-larger-than -Wno-aggressive-loop-optimizations
LDFLAGS  := -shared -llapacke -lopenblas

# Library suffix based on OS
SO_SUFFIX := .so
ifeq ($(OS),Windows_NT)
  SO_SUFFIX := .dll
endif

# Paths
CSRCDIR    := csrc
LIBDIR     := $(CSRCDIR)/build
BENCHDIR     := benchmarks
FLOPSDIR     := $(BENCHDIR)/flops


# src and targets
CSRCS      := $(wildcard $(CSRCDIR)/*.c)
TARGET_LIBS := $(patsubst $(CSRCDIR)/%.c,$(LIBDIR)/lib%$(SO_SUFFIX),$(CSRCS))
TARGET_LIBSFLOPS := $(patsubst $(CSRCDIR)/%.c,$(LIBDIR)/lib%_flops$(SO_SUFFIX),$(CSRCS))

.PHONY: all

all: libs

libs: lib libflops

lib: $(TARGET_LIBS)
$(LIBDIR)/lib%$(SO_SUFFIX): $(CSRCDIR)/%.c | $(LIBDIR)
	@echo "Building C shared library → $@"
	$(CC) $(CFLAGS) -UTRACEFLOPS $(LDFLAGS) -o $@ $<

libflops: $(TARGET_LIBSFLOPS) $(FLOPSDIR)
$(LIBDIR)/lib%_flops$(SO_SUFFIX): $(CSRCDIR)/%.c | $(LIBDIR)
	@echo "Building C shared library → $@"
	$(CC) $(CFLAGS) -DTRACEFLOPS $(LDFLAGS) -o $@ $<

$(LIBDIR):
	mkdir -p $(LIBDIR)

$(FLOPSDIR):
	mkdir -p $(FLOPSDIR)

clean:
	@echo "Cleaning up…"
	rm -rf $(LIBDIR)
	rm -rf $(FLOPSDIR)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache


