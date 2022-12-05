#include <immintrin.h>
#include <time.h>
#include <math.h>

#include "simd.h"
#include "day-5-input.c"

#define CRATE_BYTES 92

typedef struct crate_stack {
  // There are only 54 items we need to worry about,
  // but we want to give ourselves a bit more room
  // to avoid writing past the end of the array.
  uint8_t bytes[CRATE_BYTES];
  size_t size;
} crate_stack_t;

// Parse a line of the crate stacks into a 256-bit vector,
// storing the first 8 characters in the low bytes of the
// first lane, and the 9th character in the low byte of the
// second lane.
static inline __m256i parse_crate_line(uint8_t *input, uint32_t *offset) {
  // TODO: This code is broken
  const __m256i chunk = _mm256_loadu_si256((__m256i*)(input + 1 + *offset));
  // Annoyingly, this lies just outside of the range that we can
  // fit inside of a 256-bit wide vector.
  const uint8_t ninth = *(input + *offset + 33);
  *offset += 36;
  // The entire line doesn't fit in a 128-bit vector,
  // so we opt to use AVX-2 instead.
  // This means that the location of the bytes in the vector
  // are weird, as we can't shuffle between lanes.
  const __m256i shuffle =
    _mm256_setr_epi8(0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80,
		     0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
		     0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80,
		     0x80, 0x80, 0x80, 0x80, 0x80, 0x08, 0x80, 0x80);
  __m256i crates = _mm256_shuffle_epi8(chunk, shuffle);
  // Tack on the 9th crate that we had to load separately.
  __m256i ninth_crate = _mm256_set1_epi8(ninth);
  const __m256i ninth_crate_mask =
    _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0,
		     0, 0, 0, 0, 0, 0, 0, 0,
		     0, 0, 0, 0, 0xff, 0, 0, 0,
		     0, 0, 0, 0, 0, 0, 0, 0);
  crates = _mm256_blendv_epi8(crates, ninth_crate, ninth_crate_mask);
  // Replace any space characters with a 0.
  __m256i mask = _mm256_cmpeq_epi8(crates, _mm256_set1_epi8(' '));
  return _mm256_blendv_epi8(crates, _mm256_setzero_si256(), mask);
}

// Parse a single move instruction from a 128-bit vector,
// and update the value pointed to by offset by the number
// of bytes consumed.
// The output can be found in bytes 0, 2, 4, and 16.
static inline __m256i parse_move_epi8x4(uint8_t *input, uint32_t *offset) {
  const __m256i chunk = _mm256_loadu_si256((__m256i*) (input + *offset));
  // Extract the position of the newline.
  // This will be used to update the offset and
  // to determine how digits the move quantity has.
  const __m256 newline_epi8 = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\n'));
  const uint32_t newline = _mm_tzcnt_64(_mm256_movemask_epi8(newline_epi8));
  // The newline should be at either byte 19 or 18, so we subtract
  // 18 to get the offsets of the 3rd and 4th digits.
  const int32_t digit_offset = newline - 18;
  // For the 3rd and 4th bytes, we just need to add the appropriate offset.
  const __m256i digits_shuffle =
    _mm256_setr_epi8(5, 0x80, 6, 0x80,
		     digit_offset + 12, 0x80, 0x80, 0x80,
                     0x80, 0x80, 0x80, 0x80,
                     0x80, 0x80, 0x80, 0x80,
                     digit_offset + 17, 0x80, 0x80, 0x80,
                     0x80, 0x80, 0x80, 0x80,
                     0x80, 0x80, 0x80, 0x80,
                     0x80, 0x80, 0x80, 0x80);
  
  *offset += newline + 1;
  __m256i digits = _mm256_shuffle_epi8(chunk, digits_shuffle);
  // The 2nd byte we read might be a space. If it is, we
  // need to set it to 0.
  __m256i is_space = _mm256_cmpeq_epi8(digits, _mm256_set1_epi8(' '));
  digits = _mm256_blendv_epi8(digits, _mm256_set1_epi8('0'), is_space);
  digits = _mm256_sub_epi8(digits, _mm256_set1_epi8('0'));
  const __m256i scale =
    _mm256_setr_epi16(1 + 9*digit_offset, 1, 1, 1,
		      1, 1, 1, 1,
		      1, 1, 1, 1,
		      1, 1, 1, 1);
  digits = _mm256_mullo_epi16(digits, scale);
  return digits;
}

static inline void move_crates_scalar(crate_stack_t* src, crate_stack_t* dst, uint8_t count) {
  for(int i = 0; i < count; i++) {
    dst->bytes[dst->size + i] = src->bytes[src->size - count + i];
    src->bytes[src->size + i] = 0;
  }

  src->size -= count;
  dst->size += count;
}

static inline void move_crates_sse(crate_stack_t* src, crate_stack_t* dst, uint8_t count) {
  __m128i *src_ptr = (__m128i*) (src->bytes + src->size - count);
  __m128i *dst_ptr = (__m128i*) (dst->bytes + dst->size);
  __m128i bytes = _mm_loadu_si128(src_ptr);
  // Zero out the bytes read from src, and write the bytes to dst.
  _mm_storeu_si128(src_ptr, _mm_setzero_si128());
  _mm_storeu_si128(dst_ptr, bytes);
  src->size -= count;
  dst->size += count;
}

static inline void move_crates_rev_scalar(crate_stack_t* src, crate_stack_t* dst, uint8_t count) {
  for(int i = 0; i < count; i++) {
    dst->bytes[dst->size + i] = src->bytes[src->size - i - 1];
    src->bytes[src->size - i] = 0;
  }

  src->size -= count;
  dst->size += count;
}

static inline void move_crates_rev_sse(crate_stack_t* src, crate_stack_t* dst, uint8_t count) {
  __m128i *src_ptr = (__m128i*) (src->bytes + src->size - count);
  __m128i *dst_ptr = (__m128i*) (dst->bytes + dst->size);
  __m128i bytes = _mm_loadu_si128(src_ptr);
  // Load up an the sequence [0,1,2..15] into each of the lanes
  // of a 128-bit vector.
  __m128i shuffle =
    _mm_set_epi64x(0x0f0e0d0c0b0a0908, 0x0706050403020100);
  // Subtract off count to get the sequence [count, count - 1, ..., 0, -1, ...].
  shuffle = _mm_sub_epi8(_mm_set1_epi8(count - 1), shuffle);
  // Replace the negative bytes with 0x80.
  // This leaves us with [count, count - 1, ..., 0, 0x80, 0x80, ...],
  // which allows us to perform a reversal of bytes.
  const __m128i mask = _mm_cmpgt_epi8(_mm_setzero_si128(), shuffle);
  shuffle = _mm_blendv_epi8(shuffle, _mm_set1_epi8(0x80), mask);
  bytes = _mm_shuffle_epi8(bytes, shuffle);

  // Zero out the bytes read from src, and write the reversed
  // bytes to dst.
  _mm_storeu_si128(src_ptr, _mm_setzero_si128());
  _mm_storeu_si128(dst_ptr, bytes);
  src->size -= count;
  dst->size += count;
}

// This macro is wildly unhygenic, but I cannot be bothered.
#define PUSH_CRATE(n, ix)                                                      \
  do {                                                                         \
    uint8_t byte = _mm256_extract_epi8(line, ix);                              \
    crates[n].bytes[7 - i] = byte;                                             \
    crates[n].size += (byte != 0);                                             \
    crates_rev[n].bytes[7 - i] = byte;			\
    crates_rev[n].size += (byte != 0);			\
  } while (0)

#define CRATE_STACK_TOP(n) crates[n].bytes[crates[n].size - 1]
#define CRATE_REV_STACK_TOP(n) crates_rev[n].bytes[crates_rev[n].size - 1]

int main() {
  clock_t start_time = clock();

  uint8_t *input = __src_day_5_txt;
  uint32_t offset = 0;
  crate_stack_t crates[9] = {};
  crate_stack_t crates_rev[9] = {};


  // Initialize the crates.
  // We don't have any good strided scatter operations,
  // which makes this somewhat inefficient.
  uint8_t bytes[32];
  for (int i = 0; i < 8; i++) {
    const __m256i line = parse_crate_line(input, &offset);
    PUSH_CRATE(0, 0);
    PUSH_CRATE(1, 1);
    PUSH_CRATE(2, 2);
    PUSH_CRATE(3, 3);
    PUSH_CRATE(4, 16);
    PUSH_CRATE(5, 17);
    PUSH_CRATE(6, 18);
    PUSH_CRATE(7, 19);
    PUSH_CRATE(8, 20);
  }

  offset += 37;

  while(offset < __src_day_5_txt_len) {
    const __m256i move = parse_move_epi8x4(input, &offset);
    _mm256_storeu_si256((__m256i*) bytes, move);
    crate_stack_t *src = crates + bytes[4] - 1;
    crate_stack_t *dst = crates + bytes[16] - 1;
    crate_stack_t *src_rev = crates_rev + bytes[4] - 1;
    crate_stack_t *dst_rev = crates_rev + bytes[16] - 1;
    const uint32_t count = bytes[0] + bytes[2];

    // We unfortunately need to branch here, as SSE vectors
    // can't fit any more bytes.
    // We could use AVX2 256-bit wide vectors, but the
    // cross lane shuffle required for the reverse is very annoying,
    // and we'd still need to fall back to scalar ops for some cases.
    // Luckilly we don't need to fall back to the scalar impl too often,
    // so this should be friendly to the branch predictor.
    if (count <= 16) {
      move_crates_sse(src, dst, count);
      move_crates_rev_sse(src_rev, dst_rev, count);
    } else {
      move_crates_scalar(src, dst, count);
      move_crates_rev_scalar(src_rev, dst_rev, count);
    }
  }

  double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

  printf("Rev Crates: %c%c%c%c%c%c%c%c%c\n",
	 CRATE_REV_STACK_TOP(0),
	 CRATE_REV_STACK_TOP(1),
	 CRATE_REV_STACK_TOP(2),
	 CRATE_REV_STACK_TOP(3),
	 CRATE_REV_STACK_TOP(4),
	 CRATE_REV_STACK_TOP(5),
	 CRATE_REV_STACK_TOP(6),
	 CRATE_REV_STACK_TOP(7),
	 CRATE_REV_STACK_TOP(8));
  printf("Crates:     %c%c%c%c%c%c%c%c%c\n",
	 CRATE_STACK_TOP(0),
	 CRATE_STACK_TOP(1),
	 CRATE_STACK_TOP(2),
	 CRATE_STACK_TOP(3),
	 CRATE_STACK_TOP(4),
	 CRATE_STACK_TOP(5),
	 CRATE_STACK_TOP(6),
	 CRATE_STACK_TOP(7),
	 CRATE_STACK_TOP(8));
  printf("Completed in %1.0lf microseconds", elapsed_time * pow(10, 6));
}
