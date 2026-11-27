# Desktop build of the GDN bring-up demo.
#
#   make
#   ./bin/run_gdn
#   make clean
#
# This does NOT replace locod.sh; it lets you verify the CPU reference before
# putting the same kernel on the FPGA.
CC       ?= gcc
CFLAGS   ?= -O2 -Wall -Wextra
LDLIBS   :=
TARGET   := bin/run_gdn
OBJDIR   := build
OBJS     := $(OBJDIR)/main.o $(OBJDIR)/gdn.o

.PHONY: all run clean
all: $(TARGET)

$(TARGET): $(OBJS) | bin
	$(CC) $(OBJS) $(LDLIBS) -o $@

$(OBJDIR)/main.o: main.c io.h locod_host.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/gdn.o: gdn.c io.h params.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR) bin:
	mkdir -p $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(OBJDIR) bin
