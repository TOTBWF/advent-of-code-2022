#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "day-1-input.h"
#include "simd.h"

void pqueue_push(__m128i* pqueue, uint32_t x) {
  const __m128i vec = _mm_set1_epi32(x);
  const __m128i lt_mask = _mm_cmplt_epi32(*pqueue, vec);
  const int num_less = _mm_popcnt_u32(_mm_movemask_epi8(lt_mask)) / 4;
  __m128i queue_mask;
  __m128i x_pos;

  switch (num_less) {
  case 0:
    break;
  case 1:
    queue_mask = _mm_setr_epi8(0x80, 0x80, 0x80, 0x80, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    x_pos = _mm_setr_epi32(x, 0, 0, 0);
    *pqueue = _mm_or_si128(x_pos, _mm_shuffle_epi8(*pqueue, queue_mask));
    break;
  case 2:
    queue_mask = _mm_setr_epi8(4, 5, 6, 7, 0x80, 0x80, 0x80, 0x80, 8, 9, 10, 11, 12, 13, 14, 15);
    x_pos = _mm_setr_epi32(0, x, 0, 0);
    *pqueue = _mm_or_si128(x_pos, _mm_shuffle_epi8(*pqueue, queue_mask));
    break;
  case 3:
    queue_mask = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 0x80, 0x80, 0x80, 0x80, 12, 13, 14, 15);
    x_pos = _mm_setr_epi32(0, 0, x, 0);
    *pqueue = _mm_or_si128(x_pos, _mm_shuffle_epi8(*pqueue, queue_mask));
    break;
  case 4:
    queue_mask = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0x80, 0x80, 0x80, 0x80);
    x_pos = _mm_setr_epi32(0, 0, 0, x);
    *pqueue = _mm_or_si128(x_pos, _mm_shuffle_epi8(*pqueue, queue_mask));
    break;
  }

}

int pqueue_top_3(__m128i pqueue) {
  const __m128 mask = _mm_setr_epi32(0, 0xffffffff, 0xffffffff, 0xffffffff);
  const __m128 pqueue_top = _mm_and_si128(mask, pqueue);
  const __m128 a0 = _mm_hadd_epi32(pqueue_top, pqueue_top);
  return _mm_cvtsi128_si32(_mm_hadd_epi32(a0, a0));
}

int pqueue_top(__m128i pqueue) {
  return _mm_extract_epi32(pqueue, 3);
}

int main() {
  clock_t start_time = clock();

  uint8_t *input = input_day_1_txt;
  uint32_t current = 0;
  __m128i pqueue = _mm_set1_epi32(0);
  while(*input) {
    if (*input == '\n') {
      pqueue_push(&pqueue, current);
      current = 0;
      input += 1;
    } else if (input[4] == '\n') {
      __m128i chunk = _mm_loadu_si128((__m128i*)input);
      current += parse_4_digits(chunk);
      input += 5;
    } else {
      __m128i chunk = _mm_loadu_si128((__m128i*)input);
      current += parse_5_digits(chunk);
      input += 6;
    }
  }

  double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
  printf("Sum of Top  : %d\nSum of Top 3: %d\nCompleted in %1.0lf microseconds", pqueue_top(pqueue), pqueue_top_3(pqueue), elapsed_time * pow(10, 6));
  return 0;
}
