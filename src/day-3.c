#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "simd.h"
#include "day-3-input.c"


#define SHUFFLE_256_MASK(n)                                                    \
  _mm256_setr_epi64x(                                                          \
      0xffffffffffffff00 | 2 * n, 0xffffffffffffff00 | (2 * n + 1),            \
      0xffffffffffffff00 | 2 * n, 0xffffffffffffff00 | (2 * n + 1))

// Compute a bitset from a 256-bit input vector of [32 x u8] that
// denotes whether or not the bytes [a..z] | [A..Z] are contained
// in the input vector.
//
// Each 64-bit bitset is stored in 2 halves; extracting the full bitset
// requires a horizontal and across each 128-bit lane.
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
  return v + (is_upper * 27) + ((is_upper - 1) * 31);
}

static inline __m128i priority_epi32x4(__m128i v) {
  const __m128i is_upper = _mm_cmplt_epi32(v, _mm_set1_epi32(26));
  const __m128i upper_shift = _mm_mullo_epi32(is_upper, _mm_set1_epi32(-27));
  const __m128i what = _mm_add_epi32(is_upper, _mm_set1_epi32(1));
  const __m128i lower_shift = _mm_mullo_epi32(what, _mm_set1_epi32(-31));
  return _mm_add_epi32(v, _mm_add_epi32(upper_shift, lower_shift));
}

static inline void rucksack_contents(uint8_t* input, uint32_t* offset, __m128i* out) {
    const uint32_t len = _mm256_line_len(input + *offset);
    const __m256 mask = _mm256_expand_mask(0xffffffffffffffff << len/2);
    const __m256 lo_chunk =
      _mm256_andnot_si256(mask, _mm256_loadu_si256((__m256i*) (input + *offset)));
    const __m256 hi_chunk =
      _mm256_andnot_si256(mask, _mm256_loadu_si256((__m256i*) (input + *offset + len/2)));
    *out = _mm_set_epi64x(_mm256_hor_epi64(bitset_epi8x32(lo_chunk)), _mm256_hor_epi64(bitset_epi8x32(hi_chunk)));
    *offset += len + 1;
}

int main() {
  clock_t start_time = clock();

  uint32_t offset = 0;
  uint8_t* input = __src_day_3_txt;
  __m128i rearrange_priorities = _mm_setzero_si128();
  uint32_t group_priorities = 0;
  __m128i rucksack_bv;
  while(offset < __src_day_3_txt_len) {
    rucksack_contents(input, &offset, &rucksack_bv);
    const uint8_t diff0 = _tzcnt_u64(_mm_hand_epi64(rucksack_bv));
    uint64_t group = _mm_hor_epi64(rucksack_bv);

    rucksack_contents(input, &offset, &rucksack_bv);
    const uint8_t diff1 = _tzcnt_u64(_mm_hand_epi64(rucksack_bv));
    group = group & _mm_hor_epi64(rucksack_bv);

    rucksack_contents(input, &offset, &rucksack_bv);
    const uint8_t diff2 = _tzcnt_u64(_mm_hand_epi64(rucksack_bv));
    group = group & _mm_hor_epi64(rucksack_bv);

    // HACK: We use -27 here as it will get taken to priority 0 by priority_epi32x4.
    const __m128i priorities_chunk = priority_epi32x4(_mm_setr_epi32(diff0, diff1, diff2, -27));
    rearrange_priorities = _mm_add_epi32(rearrange_priorities, priorities_chunk);
    group_priorities += priority(_mm_tzcnt_64(group));
  }

  double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
  printf("Priority Sum: %d\nGroup Priority Sum: %d\nCompleted in %1.0lf microseconds", _mm_hsum_epi32(rearrange_priorities), group_priorities, elapsed_time * pow(10, 6));
}
