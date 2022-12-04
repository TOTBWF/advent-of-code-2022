#include <emmintrin.h>
#include <immintrin.h>
#include <popcntintrin.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "day-4-input.c"
#include "simd.h"

static inline __m128i extract_range_epi32x4(uint8_t *input, uint32_t *offset) {
  const __m128i chunk = _mm_loadu_si128((__m128i*) (input + *offset));
  const uint16_t dash_positions = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, _mm_set1_epi8('-')));
  const uint16_t comma_position = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, _mm_set1_epi8(',')));
  const uint16_t newline_position = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, _mm_set1_epi8('\n')));
  const uint16_t dash_lo = _mm_tzcnt_64(dash_positions);
  const uint16_t dash_hi = _mm_tzcnt_64(dash_positions >> (dash_lo + 1)) + dash_lo + 1;
  const uint16_t comma = _mm_tzcnt_64(comma_position);
  const uint16_t newline = _mm_tzcnt_64(newline_position);

  const __m128i start =
    _mm_add_epi8(_mm_set_epi8(0x7f, 0x7f, 0x7f, dash_hi,
			      0x7f, 0x7f, 0x7f, comma,
			      0x7f, 0x7f, 0x7f, dash_lo,
			      0x7f, 0x7f, 0x7f, -1),
		 _mm_set1_epi8(1));
  const __m128i end =
    _mm_sub_epi8(_mm_set_epi8(0x81, 0x81, 0x81, newline,
			      0x81, 0x81, 0x81, dash_hi,
			      0x81, 0x81, 0x81, comma,
			      0x81, 0x81, 0x81, dash_lo),
		 _mm_set1_epi8(1));

  __m128i normalized = _mm_sub_epi8(chunk, _mm_set1_epi8('0'));
  __m128i scale = _mm_mullo_epi32(_mm_set1_epi32(10), _mm_sub_epi32(end, start));
  __m128i hi = _mm_mullo_epi32(scale, _mm_shuffle_epi8(normalized, start)) ;
  __m128i lo = _mm_shuffle_epi8(normalized, end);

  *offset += newline + 1;
  return _mm_add_epi32(hi, lo);
}

int main() {
  clock_t start_time = clock();

  uint8_t *input = __src_day_4_txt;
  uint32_t offset = 0;
  uint32_t contains = 0;
  uint32_t overlaps = 0;
  while (offset < __src_day_4_txt_len) {
    __m128i range0 = extract_range_epi32x4(input, &offset);
    __m128i range1 = extract_range_epi32x4(input, &offset);
    __m128i range2 = extract_range_epi32x4(input, &offset);
    __m128i range3 = extract_range_epi32x4(input, &offset);

    _MM_TRANSPOSE4_PS(range0, range1, range2, range3);

    const __m128i lo_lo_lt = _mm_cmplt_epi32(range0, range2);
    const __m128i lo_lo_eq = _mm_cmpeq_epi32(range0, range2);
    const __m128i lo_lo_gt = _mm_cmpgt_epi32(range0, range2);
    const __m128i lo_lo_le = _mm_or_si128(lo_lo_eq, lo_lo_lt);
    const __m128i lo_lo_ge = _mm_or_si128(lo_lo_eq, lo_lo_gt);

    const __m128i hi_hi_lt = _mm_cmplt_epi32(range1, range3);
    const __m128i hi_hi_eq = _mm_cmpeq_epi32(range1, range3);
    const __m128i hi_hi_gt = _mm_cmpgt_epi32(range1, range3);
    const __m128i hi_hi_le = _mm_or_si128(hi_hi_eq, hi_hi_lt);
    const __m128i hi_hi_ge = _mm_or_si128(hi_hi_eq, hi_hi_gt);

    const __m128i lo_hi_lt = _mm_cmplt_epi32(range0, range3);
    const __m128i lo_hi_eq = _mm_cmpeq_epi32(range0, range3);
    const __m128i lo_hi_gt = _mm_cmpgt_epi32(range0, range3);
    const __m128i lo_hi_le = _mm_or_si128(lo_hi_eq, lo_hi_lt);
    const __m128i lo_hi_ge = _mm_or_si128(lo_hi_eq, lo_hi_gt);

    const __m128i hi_lo_lt = _mm_cmplt_epi32(range1, range2);
    const __m128i hi_lo_eq = _mm_cmpeq_epi32(range1, range2);
    const __m128i hi_lo_gt = _mm_cmpgt_epi32(range1, range2);
    const __m128i hi_lo_le = _mm_or_si128(hi_lo_eq, hi_lo_lt);
    const __m128i hi_lo_ge = _mm_or_si128(hi_lo_eq, hi_lo_gt);

    const __m128i contains_bv = _mm_or_si128(_mm_and_si128(lo_lo_le, hi_hi_ge),
					     _mm_and_si128(lo_lo_ge, hi_hi_le));
    const __m128i overlaps_bv = _mm_or_si128(_mm_and_si128(lo_hi_le, hi_lo_ge),
					     _mm_and_si128(lo_hi_ge, hi_hi_le));

    contains += _mm_popcnt_u32(_mm_movemask_epi8(contains_bv)) / 4;
    overlaps += _mm_popcnt_u32(_mm_movemask_epi8(overlaps_bv)) / 4;
  }

  double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
  printf("Contains: %d\nOverlaps: %d\nCompleted in %1.0lf microseconds", contains, overlaps, elapsed_time * pow(10, 6));
  return 0;
}
