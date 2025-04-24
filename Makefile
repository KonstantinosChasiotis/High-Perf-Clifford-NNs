# Makefile

# Compiler and flags
CC       := gcc
CFLAGS   := -O3 -fPIC
LDFLAGS  := -shared

# Library name (shared object)
SO_NAME  := libcli2ord.so
ifeq ($(OS),Windows_NT)
  SO_NAME := cli2ord_linear.dll
endif

# Paths
CSRCDIR    := csrc
LIBDIR     := $(CSRCDIR)
LIBPATH    := $(LIBDIR)/$(SO_NAME)
SRCS       := $(CSRCDIR)/clifford_linear.c
HDRS       := 

# Pytest command
PYTEST := pytest --maxfail=1 --disable-warnings -q

.PHONY: all test clean

all: $(LIBPATH)

$(LIBPATH): $(SRCS) $(HDRS) | $(LIBDIR)
	@echo "Building C shared library → $@"
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

$(LIBDIR):
	mkdir -p $(LIBDIR)

test: all
	@echo "Running Python tests"
	$(PYTEST)

clean:
	@echo "Cleaning up…"
	rm -f $(LIBPATH)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache

