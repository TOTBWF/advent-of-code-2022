#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <tmmintrin.h>

#include "day-4-input.c"
#include "simd.h"

static inline __m128i extract_range_epi32x4(uint8_t *input, uint32_t *offset) {
  const __m128i chunk = _mm_loadu_si128((__m128i*) (input + *offset));
  const __m128i symbols = _mm_cmplt_epi8(chunk, _mm_set1_epi8('0'));

  // Compress down the mask to a 16x4i.
  __m128i mask = _mm_srli_epi64(symbols, 4);
  const __m128i compress_mask =
    _mm_set_epi8(0x80, 0x80, 0x80, 0x80,
		 0x80, 0x80, 0x80, 0x80,
		 0x0E, 0x0C, 0x0A, 0x08,
		 0x06, 0x04, 0x02, 0x00);
  mask = _mm_shuffle_epi8(mask, compress_mask);
  const uint64_t mask64 = _mm_cvtsi128_si64(mask);
  // Extract the relevant 4-bit indicies, and store them in the low bits.
  __m128i indicies = _mm_cvtsi64_si128(_pext_u64(0xFEDCBA9876543210, mask64));

  indicies = _mm_unpacklo_epi8(indicies, _mm_srli_epi64(indicies, 4));
  const __m128i indicies_mask =
    _mm_setr_epi8(0x0F, 0x0F, 0x0F, 0x0F,
		  0x00, 0x00, 0x00, 0x00,
		  0x00, 0x00, 0x00, 0x00,
		  0x00, 0x00, 0x00, 0x00);
  indicies = _mm_and_si128(indicies, indicies_mask);
  /* Store on magic values. */
  const __m128i magic =
    _mm_setr_epi8(0x00, 0x00, 0x00, 0x00,
		  0xFF, 0x7f, 0x81, 0x00,
		  0x00, 0x00, 0x00, 0x00,
		  0x00, 0x00, 0x00, 0x00);
  indicies = _mm_or_si128(indicies, magic);

  const __m128i start_shuffle =
    _mm_set_epi8(5, 5, 5, 2,
		 5, 5, 5, 1,
		 5, 5, 5, 0,
		 5, 5, 5, 4);
  const __m128i start = _mm_add_epi8(_mm_shuffle_epi8(indicies, start_shuffle), _mm_set1_epi8(1));
  const __m128i end_shuffle =
    _mm_set_epi8(6, 6, 6, 3,
		 6, 6, 6, 2,
		 6, 6, 6, 1,
		 6, 6, 6, 0);
  const __m128i end = _mm_sub_epi8(_mm_shuffle_epi8(indicies, end_shuffle), _mm_set1_epi8(1));

  __m128i normalized = _mm_sub_epi8(chunk, _mm_set1_epi8('0'));
  __m128i scale = _mm_mullo_epi32(_mm_set1_epi32(10), _mm_sub_epi32(end, start));
  __m128i hi = _mm_mullo_epi32(scale, _mm_shuffle_epi8(normalized, start)) ;
  __m128i lo = _mm_shuffle_epi8(normalized, end);

  *offset += _mm_extract_epi8(indicies, 3) + 1;
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

    // NOTE: We don't have _mm_cmple_epi32 until AVX-512, so
    // we have to emulate it by doing strict comparisons + equality tests.
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
