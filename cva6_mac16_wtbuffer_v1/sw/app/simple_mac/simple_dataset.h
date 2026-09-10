#pragma once
#include <stdint.h>

#define DATA_SIZE 10

static const uint8_t input1_data[DATA_SIZE] __attribute__((aligned(4))) = {
  8, 4, 6, 5, 8, 1, 6, 0, 7, 6
};

static const int8_t input2_data[DATA_SIZE] __attribute__((aligned(4))) = {
  4, -5, 1, 9, -3, 4, -4, 0, 7, 6
};
