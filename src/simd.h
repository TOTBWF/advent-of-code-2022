#include <emmintrin.h>
#include <immintrin.h>
#include <stdio.h>

// https://stackoverflow.com/a/60109639
static inline uint32_t _mm_hsum_epi32(__m128i x) {
    const __m128i hi64  = _mm_unpackhi_epi64(x, x);
    const __m128i sum64 = _mm_add_epi32(hi64, x);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

static inline uint32_t _mm256_hsum_epi32(__m256i x) {
      __m128i sum128 = _mm_add_epi32( 
                 _mm256_castsi256_si128(x),
                 _mm256_extracti128_si256(x, 1));
    return _mm_hsum_epi32(sum128);
}

static inline uint64_t _mm_hor_epi64(__m128i x) {
  const __m128i hi64 = _mm_unpackhi_epi64(x, x);
  const __m128i lo64 = _mm_unpacklo_epi64(x, x);
  const __m128i or64 = _mm_or_si128(hi64, lo64);
  return _mm_cvtsi128_si64(or64);
}

static inline uint64_t _mm256_hor_epi64(__m256i x) {
  __m128i or128 = _mm_or_si128(_mm256_castsi256_si128(x),
			       _mm256_extracti128_si256(x, 1));
  return _mm_hor_epi64(or128);
}

static inline uint64_t _mm_hand_epi64(__m128i x) {
  const __m128i hi64 = _mm_unpackhi_epi64(x, x);
  const __m128i lo64 = _mm_unpacklo_epi64(x, x);
  const __m128i or64 = _mm_and_si128(hi64, lo64);
  return _mm_cvtsi128_si64(or64);
}

static inline uint32_t _mm256_line_len(uint8_t *str) {
  __m256i newline = _mm256_set1_epi8('\n');
  int32_t i = 0;
  int32_t mask = 0;
  while (1) {
    __m256i chunk = _mm256_loadu_si256((__m256i*)(str + i));
    mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, newline));
    if (mask) {
      return _tzcnt_u32(mask) + i;
    }
    i += 32;
  }
}

static inline __m256i _mm256_expand_mask(uint32_t mask) {
  __m256i vmask = _mm256_set1_epi32(mask);
  const __m256i shuffle =
    _mm256_setr_epi64x(0x0000000000000000,
		       0x0101010101010101,
		       0x0202020202020202,
		       0x0303030303030303);
  vmask = _mm256_shuffle_epi8(vmask, shuffle);
  const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
  vmask = _mm256_or_si256(vmask, bit_mask);
  return _mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1));
}

////////////////////////////////////////////////////////////////////////////////
// Debugging Functions

void print_hex_epi16x8(__m128i v) {
  printf("0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x\n",
	 _mm_extract_epi16(v, 0),
	 _mm_extract_epi16(v, 1),
	 _mm_extract_epi16(v, 2),
	 _mm_extract_epi16(v, 3),
	 _mm_extract_epi16(v, 4),
	 _mm_extract_epi16(v, 5),
	 _mm_extract_epi16(v, 6),
	 _mm_extract_epi16(v, 7));
}

void print_hex_epi32x4(__m128i v) {
  printf("0x%x 0x%x 0x%x 0x%x\n",
	 _mm_extract_epi32(v, 0),
	 _mm_extract_epi32(v, 1),
	 _mm_extract_epi32(v, 2),
	 _mm_extract_epi32(v, 3));
}

void print_epi32x4(__m128i v) {
  printf("%d %d %d %d\n",
	 _mm_extract_epi32(v, 0),
	 _mm_extract_epi32(v, 1),
	 _mm_extract_epi32(v, 2),
	 _mm_extract_epi32(v, 3));
}
