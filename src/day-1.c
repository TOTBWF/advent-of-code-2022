#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "day-1-input.h"

uint32_t parse_4_digits(const __m128i input) {
  const __m128i char_0 = _mm_set1_epi8('0');

  // Normalize the '0' char to actually be 0x00.
  const __m128i normalized = _mm_subs_epi8(input, char_0);
  // The parsing algorithm proceeds by performing 2 multiplication + adjacent add operations.
  // Our 4 digit string "1234" will get normalized to the vector [1,2,3,4]
  // The first maddubs with mul_10 will yield [12,34], and the second 1234.
  // Note that we need to convert to signed ints to be able to call _mm_cvtsi128_si32
  const __m128i mul_10 = _mm_setr_epi8(10, 1, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  const __m128i mul_100 = _mm_setr_epi16(100, 1, 0, 0, 0, 0, 0, 0);
  const __m128i digits =_mm_madd_epi16(_mm_maddubs_epi16(normalized, mul_10), mul_100);

  return _mm_cvtsi128_si32(digits);
}

uint32_t parse_5_digits(const __m128i input) {
  const __m128i char_0 = _mm_set1_epi8('0');

  // Normalize the '0' char to actually be 0x00.
  const __m128i normalized = _mm_subs_epi8(input, char_0);

  // We need to shuffle the 5th digit to be the LSB.
  const __m128i shuffle_mask = _mm_setr_epi8(0, 1, 2, 3, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
  // Same algorithm as the 4 digit case, making sure to fold in the 5th digit.
  const __m128i shuffled = _mm_shuffle_epi8(normalized, shuffle_mask);
  const __m128i mul_10 = _mm_setr_epi8(10, 1, 10, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  // We multiply the 4 top digits by an extra 10 here to account for the 5th digit.
  const __m128i mul_100 = _mm_setr_epi16(1000, 10, 0, 1, 0, 0, 0, 0);
  const __m128i digits_with_trailing = _mm_madd_epi16(_mm_maddubs_epi16(shuffled, mul_10), mul_100);
  // Add together the upper 4 digits with the bottom 5th.
  const __m128i digits = _mm_hadd_epi32(digits_with_trailing, digits_with_trailing);

  return _mm_cvtsi128_si32(digits);
}

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
