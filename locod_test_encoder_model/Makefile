# Desktop build of the LoCod encoder-model test.
#
#   make            -> bin/run_encoder
#   ./bin/run_encoder
#   make clean
#
# This does NOT replace locod.sh. It only lets you check the model on your
# laptop. The cross-compiled / board executable is still produced by:
#
#   cd ../LoCod
#   bash ./locod.sh -t pynqz2 --no-hard \
#     -f ../locod_test_encoder_model/main.c \
#        ../locod_test_encoder_model/accel.c \
#        ../locod_test_encoder_model/io.h \
#        ../locod_test_encoder_model/params.h

CC       ?= gcc
CFLAGS   ?= -O2 -Wall -Wextra
LDLIBS   := -lm

TARGET   := bin/run_encoder
OBJDIR   := build
OBJS     := $(OBJDIR)/main.o $(OBJDIR)/accel.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) | bin
	$(CC) $(OBJS) $(LDLIBS) -o $@

$(OBJDIR)/main.o: main.c io.h locod_host.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/accel.o: accel.c io.h params.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR) bin:
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) bin
