#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "simd.h"
#include "day-3-input.c"


#define SHUFFLE_256_MASK(n)                                                    \
  _mm256_setr_epi64x(                                                          \
      0xffffffffffffff00 | 2 * n, 0xffffffffffffff00 | (2 * n + 1),            \
      0xffffffffffffff00 | 2 * n, 0xffffffffffffff00 | (2 * n + 1))

__m256i bitset_epi8x32(__m256i input) {
  __m256i chunk = _mm256_sub_epi8(input, _mm256_set1_epi8(0x41));

  __m256i shuffle_mask_0 = SHUFFLE_256_MASK(0);
  __m256i shuffle_mask_1 = SHUFFLE_256_MASK(1);
  __m256i shuffle_mask_2 = SHUFFLE_256_MASK(2);
  __m256i shuffle_mask_3 = SHUFFLE_256_MASK(3);
  __m256i shuffle_mask_4 = SHUFFLE_256_MASK(4);
  __m256i shuffle_mask_5 = SHUFFLE_256_MASK(5);
  __m256i shuffle_mask_6 = SHUFFLE_256_MASK(6);
  __m256i shuffle_mask_7 = SHUFFLE_256_MASK(7);

  const __m256i chunk0 = _mm256_shuffle_epi8(chunk, shuffle_mask_0);
  const __m256i chunk1 = _mm256_shuffle_epi8(chunk, shuffle_mask_1);
  const __m256i chunk2 = _mm256_shuffle_epi8(chunk, shuffle_mask_2);
  const __m256i chunk3 = _mm256_shuffle_epi8(chunk, shuffle_mask_3);
  const __m256i chunk4 = _mm256_shuffle_epi8(chunk, shuffle_mask_4);
  const __m256i chunk5 = _mm256_shuffle_epi8(chunk, shuffle_mask_5);
  const __m256i chunk6 = _mm256_shuffle_epi8(chunk, shuffle_mask_6);
  const __m256i chunk7 = _mm256_shuffle_epi8(chunk, shuffle_mask_7);

  const __m256i ones = _mm256_set1_epi64x(1);
  const __m256 bv0 = _mm256_sllv_epi64(ones, chunk0);
  const __m256 bv1 = _mm256_sllv_epi64(ones, chunk1);
  const __m256 bv2 = _mm256_sllv_epi64(ones, chunk2);
  const __m256 bv3 = _mm256_sllv_epi64(ones, chunk3);
  const __m256 bv4 = _mm256_sllv_epi64(ones, chunk4);
  const __m256 bv5 = _mm256_sllv_epi64(ones, chunk5);
  const __m256 bv6 = _mm256_sllv_epi64(ones, chunk6);
  const __m256 bv7 = _mm256_sllv_epi64(ones, chunk7);

  const __m256 or01 = _mm256_or_si256(bv0, bv1);
  const __m256 or23 = _mm256_or_si256(bv2, bv3);
  const __m256 or45 = _mm256_or_si256(bv4, bv5);
  const __m256 or67 = _mm256_or_si256(bv6, bv7);

  const __m256i or0123 = _mm256_or_si256(or01, or23);
  const __m256i or4567 = _mm256_or_si256(or45, or67);

  return _mm256_or_si256(or0123, or4567);
}

static inline uint32_t priority(uint8_t v) {
  uint32_t is_upper = v < 26;
  return v + (is_upper * 27) - (!is_upper * 31);
}

int main() {
  clock_t start_time = clock();

  uint32_t offset = 0;
  uint8_t* input = __src_day_3_txt;
  uint32_t total = 0;
  while(offset < __src_day_3_txt_len) {
    const uint32_t len = _mm256_line_len(input + offset);
    const __m256 mask = _mm256_expand_mask(0xffffffffffffffff << len/2);
    const __m256 lo_chunk =
      _mm256_andnot_si256(mask, _mm256_loadu_si256((__m256i*) (input + offset)));
    const __m256 hi_chunk =
      _mm256_andnot_si256(mask, _mm256_loadu_si256((__m256i*) (input + offset + len/2)));
    const uint64_t lo = _mm256_hor_epi64(bitset_epi8x32(lo_chunk));
    const uint64_t hi = _mm256_hor_epi64(bitset_epi8x32(hi_chunk));
    const char diff = _tzcnt_u64(lo & hi);
    total += priority(diff);
    offset += len + 1;
  }

  double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
  printf("Priority Sum: %d\nCompleted in %1.0lf microseconds", total, elapsed_time * pow(10, 6));
}
