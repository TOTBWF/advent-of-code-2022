#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "day-6-input.h"

////////////////////////////////////////////////////////////////////////////////
// Day 6
//
// The general algorithm is as follows:
// First, load up a 16x8i vector from the input.
// We then perform 3 shift + equality test combos, and or the results
// together. Visually, this looks something like this:
//
//    m j r j m m w j m j f j h w r t
//    j r j m m w j m j f j h w r t
//    r j m m w j m j f j h w r t
//    j m m w j m j f j h w r t
//    -------------------------------
//    1 0 1 1 0 0 1 0 1 0 1 1 1
//
// Now, we need to look for a run of 4 1 bits.
// We can do this with more shifts + ors.
//
//    1 0 1 1 0 0 1 0 1 0 1 1 1 1 1 1
//    0 1 1 0 0 1 0 1 0 1 1 1 1 1 1
//    1 1 0 0 1 0 1 0 1 1 1 1 1 1
//    1 0 0 1 0 1 0 1 1 1 1 1 1
//    -------------------------------
//    0 0 0 0 0 0 0 0 0 0 1 1 1
//
// The position of the first 1 bit is then the start of the
// sequence of 4 distinct bytes.

#define SHIFT_CHUNK(chunk, shifted, mask)					\
  do {                                                                         \
    shifted = _mm_srli_si128(shifted, 1);                                      \
    mask = _mm_or_si128(mask, _mm_cmpeq_epi8(chunk, shifted));                 \
  } while (0)

#define SHIFT_MASK(mask, shifted_mask)                                         \
  do {                                                                         \
    shifted_mask = _mm_srli_si128(shifted_mask, 1);                            \
    mask = _mm_or_si128(mask, shifted_mask);                                   \
  } while (0)

static inline uint32_t find_pkt_start(uint8_t* input, size_t len) {
  uint32_t offset = 0;
  while (offset < len) {
    __m128i chunk = _mm_loadu_si128((__m128i*) (input + offset));
    __m128i shifted = chunk;
    __m128i mask = _mm_setzero_si128();

    // Perform the 3 shifts + equality tests to create a mask
    // that describes if there is a duplicate byte
    // in the next 3 bytes.
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);

    // Shift and or the mask together to find runs of 4
    // distinct elements.
    __m128i shifted_mask = mask;
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);

    // The equality tests will give us 0xFF when there
    // is a duplicate, so we need to invert the mask.
    mask = _mm_andnot_si128(mask, _mm_set1_epi8(0xFF));

    // Lop off the trailing 3 bytes, as they are total
    // garbage.
    __m128i trim_trailing_3 =
      _mm_setr_epi8(0xFF, 0xFF, 0xFF, 0xFF,
		    0xFF, 0xFF, 0xFF, 0xFF,
		    0xFF, 0xFF, 0xFF, 0xFF,
		    0xFF, 0x00, 0x00, 0x00);

    mask = _mm_and_si128(mask, trim_trailing_3);
    // Find the index of the first set byte (if there is one).
    int32_t index = _mm_tzcnt_32(_mm_movemask_epi8(mask));
    // This branch is going to be extremely friendly on the branch
    // predictor, as we only ever take it once.
    if(index < 32) {
      return index + offset + 4;
    }

    // We increment the offset by 13, as we don't know if the
    // trailing 3 bits are unique.
    offset += 13;
  }

  return -1;
}

static inline uint32_t find_msg_start(uint8_t* input, size_t len) {
  // This is basically the same algorithm as the first part,
  // just with way more shifts.
  uint32_t offset = 0;
  uint32_t msg_start = 0;
  while (offset < len) {
    // Load up a 128 byte chunk
    __m128i chunk = _mm_loadu_si128((__m128i*) (input + offset));
    __m128i shifted = chunk;
    __m128i mask = _mm_setzero_si128();

    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);
    SHIFT_CHUNK(chunk, shifted, mask);

    __m128i shifted_mask = mask;
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);
    SHIFT_MASK(mask, shifted_mask);

    mask = _mm_andnot_si128(mask, _mm_set1_epi8(0xFF));

    __m128i trim_trailing_13 =
      _mm_setr_epi8(0xFF, 0xFF, 0xFF, 0x00,
		    0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00);

    mask = _mm_and_si128(mask, trim_trailing_13);
    int32_t index = _mm_tzcnt_32(_mm_movemask_epi8(mask));
    if(index < 32) {
      return index + offset + 14;
    }

    offset += 3;
  }

  return -1;
}

int main() {
  // Part 1
  printf("==========[Part 1]==========\n");
  clock_t part_1_start = clock();
  uint32_t pkt_start = find_pkt_start(input_day_6_txt, input_day_6_txt_len);
  double part_1_time = (double)(clock() - part_1_start) / CLOCKS_PER_SEC;
  printf("%d\n", pkt_start);
  printf("Completed in %1.0lf microseconds\n", part_1_time * pow(10, 6));
  printf("==========[Part 2]==========\n");
  clock_t part_2_start = clock();
  uint32_t msg_start = find_msg_start(input_day_6_txt, input_day_6_txt_len);
  double part_2_time = (double)(clock() - part_2_start) / CLOCKS_PER_SEC;
  printf("%d\n", msg_start);
  printf("Completed in %1.0lf microseconds\n", part_2_time * pow(10, 6));
}
