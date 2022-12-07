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

static inline uint32_t _mm_hmin_epi32(__m128i x) {
    const __m128i hi64  = _mm_unpackhi_epi64(x, x);
    const __m128i min64 = _mm_min_epi32(hi64, x);
    const __m128i hi32  = _mm_shuffle_epi32(min64, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i min32 = _mm_min_epi32(min64, hi32);
    return _mm_cvtsi128_si32(min32);
}

static inline uint32_t _mm256_hmin_epi32(__m256i x) {
  __m128i min128 = _mm_min_epi32(_mm256_castsi256_si128(x),
				 _mm256_extracti128_si256(x, 1));
  return _mm_hmin_epi32(min128);
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
// Parsing

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

uint32_t parse_6_digits(const __m128i input) {
  const __m128i char_0 = _mm_set1_epi8('0');

  // Normalize the '0' char to actually be 0x00.
  const __m128i normalized = _mm_subs_epi8(input, char_0);

  // We need to shuffle the 5th and 6th digits to be the LSB.
  const __m128i shuffle_mask = _mm_setr_epi8(0, 1, 2, 3, 0x80, 0x80, 4, 5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
  // Same algorithm as the 4 digit case, making sure to fold in the 5th and 6th digits.
  const __m128i shuffled = _mm_shuffle_epi8(normalized, shuffle_mask);
  const __m128i mul_10 = _mm_setr_epi8(10, 1, 10, 1, 0, 0, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  // We multiply the 4 top digits by an extra 10 here to account for the 5th digit.
  const __m128i mul_100 = _mm_setr_epi16(10000, 100, 0, 1, 0, 0, 0, 0);
  const __m128i digits_with_trailing = _mm_madd_epi16(_mm_maddubs_epi16(shuffled, mul_10), mul_100);
  // Add together the upper 4 digits with the bottom 5th and 6th.
  const __m128i digits = _mm_hadd_epi32(digits_with_trailing, digits_with_trailing);

  return _mm_cvtsi128_si32(digits);
}

////////////////////////////////////////////////////////////////////////////////
// Debugging Functions

void print_hex_epi8x16(__m128i v) {
  printf("0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x\n",
	 _mm_extract_epi8(v, 0),
	 _mm_extract_epi8(v, 1),
	 _mm_extract_epi8(v, 2),
	 _mm_extract_epi8(v, 3),
	 _mm_extract_epi8(v, 4),
	 _mm_extract_epi8(v, 5),
	 _mm_extract_epi8(v, 6),
	 _mm_extract_epi8(v, 7),
	 _mm_extract_epi8(v, 8),
	 _mm_extract_epi8(v, 9),
	 _mm_extract_epi8(v, 10),
	 _mm_extract_epi8(v, 11),
	 _mm_extract_epi8(v, 12),
	 _mm_extract_epi8(v, 13),
	 _mm_extract_epi8(v, 14),
	 _mm_extract_epi8(v, 15));
}

void print_epi8x16(__m128i v) {
  printf("%c %c %c %c %c %c %c %c %c %c %c %c %c %c %c %c\n",
	 _mm_extract_epi8(v, 0),
	 _mm_extract_epi8(v, 1),
	 _mm_extract_epi8(v, 2),
	 _mm_extract_epi8(v, 3),
	 _mm_extract_epi8(v, 4),
	 _mm_extract_epi8(v, 5),
	 _mm_extract_epi8(v, 6),
	 _mm_extract_epi8(v, 7),
	 _mm_extract_epi8(v, 8),
	 _mm_extract_epi8(v, 9),
	 _mm_extract_epi8(v, 10),
	 _mm_extract_epi8(v, 11),
	 _mm_extract_epi8(v, 12),
	 _mm_extract_epi8(v, 13),
	 _mm_extract_epi8(v, 14),
	 _mm_extract_epi8(v, 15));
}

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

void print_32x8i(__m256i v) {
  char bytes[32];
  _mm256_storeu_si256((__m256i*) bytes, v);
  printf("%.32s\n", bytes);
}

void print_hex_32x8i(__m256i v) {
  uint8_t bytes[32];
  _mm256_storeu_si256((__m256i*) bytes, v);
  for(int i = 0; i < 32; i++) {
    printf("0x%02x ", (uint32_t)(bytes[i] & 0xFF));
  }
  printf("\n");
}

void print_8x32i(__m256i v) {
  int32_t ints[8];
  _mm256_storeu_si256((__m256i*) ints, v);
  for(int i = 0; i < 8; i++) {
    printf("%d ", ints[i]);
  }
  printf("\n");
}
