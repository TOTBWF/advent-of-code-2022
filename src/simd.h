#include <immintrin.h>

// https://stackoverflow.com/a/60109639
static inline uint32_t _mm_hsum_epi32(__m128i x) {
    __m128i hi64  = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

static inline uint32_t _mm256_hsum_epi32(__m256i x) {
      __m128i sum128 = _mm_add_epi32( 
                 _mm256_castsi256_si128(x),
                 _mm256_extracti128_si256(x, 1));
    return _mm_hsum_epi32(sum128);
}
