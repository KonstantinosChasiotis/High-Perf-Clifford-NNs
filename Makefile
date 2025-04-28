# Makefile

# Compiler and flags
CC       := gcc
CFLAGS   := -O3 -fPIC -Wno-alloc-size-larger-than
LDFLAGS  := -shared -llapacke -lopenblas

# Library suffix based on OS
SO_SUFFIX := .so
ifeq ($(OS),Windows_NT)
  SO_SUFFIX := .dll
endif

# Paths
CSRCDIR    := csrc
LIBDIR     := $(CSRCDIR)/build

# Find all C source files
CSRCS      := $(wildcard $(CSRCDIR)/*.c)
TARGET_LIBS := $(patsubst $(CSRCDIR)/%.c,$(LIBDIR)/lib%$(SO_SUFFIX),$(CSRCS))

.PHONY: all test clean

# Default target: Build all libraries
all: $(TARGET_LIBS)

$(LIBDIR)/lib%$(SO_SUFFIX): $(CSRCDIR)/%.c | $(LIBDIR)
	@echo "Building C shared library → $@"
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

$(LIBDIR):
	mkdir -p $(LIBDIR)

clean:
	@echo "Cleaning up…"
	rm -rf $(LIBDIR)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache

