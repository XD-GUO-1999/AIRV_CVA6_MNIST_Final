#include <stdint.h>
#include "simple_dataset.h"

int mac_c(int n, const uint8_t *a, const int8_t *b) {
  int32_t res = 0;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
  }
  return (int)res;
}

int mac_asm(int n, const uint8_t *a, const int8_t *b)
{
  int32_t res = 0;

  for (int i = 0; i < n/2; i++) {
    const uint8_t *ap = a + 2*i;
    const int8_t  *bp = b + 2*i;

    asm volatile (
      "lbu  t1, 0(%[ap])\n\t"
      "lb   t2, 0(%[bp])\n\t"
      "mul  t3, t1, t2\n\t"
      "add  %[res], %[res], t3\n\t"
      "lbu  t1, 1(%[ap])\n\t"
      "lb   t2, 1(%[bp])\n\t"
      "mul  t3, t1, t2\n\t"
      "add  %[res], %[res], t3\n\t"
      : [res] "+r" (res)
      : [ap] "r" (ap), [bp] "r" (bp)
      : "t1", "t2", "t3", "memory"
    );
  }

  if (n & 1) {
    res += a[n - 1] * b[n - 1];
  }

  return (int)res;
}

/*
int mac_asm2(int n, const uint8_t *a, const int8_t *b)
{
  int32_t res = 0;

  for (int i = 0; i < n/2; i++) {
    const uint8_t *ap = a + 2*i;
    const int8_t  *bp = b + 2*i;

    asm volatile (
      "..."
      "..."
      "..."
      "..."
      "..."
      "..."
      "..."
      "..."
      : [res] "+r" (res)
      : [ap] "r" (ap), [bp] "r" (bp)
      : "t1", "t2", "t3", "t4", "t5", "t6", "memory"
    );
  }

  if (n & 1) {
    res += a[n - 1] * b[n - 1];
  }

  return (int)res;
}*/

/* Modified version of the simple mac function: Accelerated version with Customized SIMD Instruction */
int mac_asm_simd(int n, const uint8_t *a, const int8_t *b)
{
  int32_t res = 0;

  for (int i = 0; i < n/4; i++) {
    const uint8_t *ap = a + 4*i;
    const int8_t  *bp = b + 4*i;

    asm volatile (
      "lw t1, 0(%[ap])\n\t"
      "lw t2, 0(%[ap])\n\t"
      "mac4 %[res], t1, t2\n\t"
      : [res] "+r" (res)
      : [ap] "r" (ap), [bp] "r" (bp)
      : "t1", "t2", "memory"
    );
  }

  int r = n & 3;        // non processed elements (0..3), if any 
  int k = n - r;        // index of first element non processed
  for (int j = 0; j < r; j++) {
    res += a[k + j] * b[k + j];
  }

  return (int)res;
}

int main()
{
  int sum = 0;
  //sum = mac_c(DATA_SIZE, input1_data, input2_data);
  //sum = mac_asm(DATA_SIZE, input1_data, input2_data);
  //sum = mac_asm2(DATA_SIZE, input1_data, input2_data);
  sum = mac_asm_simd(DATA_SIZE, input1_data, input2_data);
  return sum == 104;
}
