SRC_DIR := ./mnist

OBJ_FILES := $(patsubst %.c,%.o,$(wildcard $(SRC_DIR)/*.c))

CC := /usr/bin/gcc

CFLAGS := -Wall -O0 -pg -g -fno-inline \
          -I$(SRC_DIR) \
          -I./common

LDFLAGS := -pg -g

TARGET := mnist_app

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(CC) $(LDFLAGS) -o $@ $^

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ_FILES) $(TARGET) gmon.out mnist_profile.txt