#include <immintrin.h>
#include <stdio.h>

#include "day-2-input.c"

#define INVALID 0

static inline __m128i create_index(__m128i chunk) {
  // Extract out the relevant bytes, ignoring the spaces and newlines.
  // We place these bytes in the LSB of a 32-bit int to make the reduction step easier.
  const __m128i player_1_mask = _mm_setr_epi8(0, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 8, 0x80, 0x80, 0x80, 12, 0x80, 0x80, 0x80);
  const __m128i player_2_mask = _mm_setr_epi8(2, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80, 14, 0x80, 0x80, 0x80);
  // We now need to construct an index into our score lookup table.
  // The general plan is to combine together bits from the player 1 bytes and the player 2 bytes
  // to construct a 4 bit pattern, which will then be used as a mask in _mm_shuffle_epi8 to
  // perform the actual lookup. We will do this by taking the low 2 bits of player 1, shifting them
  // right, and then or-ing them with the low 2 bits of player 1.
  const __m128i player_1 = _mm_slli_epi32(_mm_shuffle_epi8(chunk, player_1_mask), 2);
  const __m128i player_2 = _mm_and_si128(_mm_shuffle_epi8(chunk, player_2_mask), _mm_set1_epi8(0x03));
  return _mm_or_si128(player_1, player_2);
  
}

int32_t lookup_score(__m128i index) {
  // For player 1, the relevant characters are:
  // 'A' (0x41)
  // 'B' (0x42)
  // 'C' (0x43)
  // For player 2, the relevant characters are:
  // 'X' (0x58)
  // 'Y' (0x59)
  // 'Z' (0x5a)
  const __m128i table =
    _mm_setr_epi8(INVALID, // 0b0000
		  INVALID, // 0b0001
		  INVALID, // 0b0010
		  INVALID, // 0b0011
		  3 + 1,   // 0b0100 (Rock/Rock)
		  6 + 2,   // 0b0101 (Rock/Paper)
		  0 + 3,   // 0b0110 (Rock/Scissors)
		  INVALID, // 0b0111
		  0 + 1,   // 0b1000 (Paper/Rock)
		  3 + 2,   // 0b1001 (Paper/Paper)
		  6 + 3,   // 0b1010 (Paper/Scissors)
		  INVALID, // 0b1011
		  6 + 1,   // 0b1100 (Scissors/Rock)
		  0 + 2,   // 0b1101 (Scissors/Paper)
		  3 + 3,   // 0b1110 (Scissors/Scissors)
		  INVALID  // 0b1111 
		  );
  const __m128i scores = _mm_shuffle_epi8(table, index);
  const __m128i total_interm = _mm_hadd_epi32(scores, scores);
  const __m128i total = _mm_hadd_epi32(total_interm, total_interm);
  return _mm_cvtsi128_si32(total);
}

int32_t lookup_outcome(__m128i index) {
  const __m128i table =
    _mm_setr_epi8(INVALID, // 0b0000
		  INVALID, // 0b0001
		  INVALID, // 0b0010
		  INVALID, // 0b0011
		  0 + 3,   // 0b0100 (Rock/Lose)
		  3 + 1,   // 0b0101 (Rock/Draw)
		  6 + 2,   // 0b0110 (Rock/Win)
		  INVALID, // 0b0111
		  0 + 1,   // 0b1000 (Paper/Lose)
		  3 + 2,   // 0b1001 (Paper/Draw)
		  6 + 3,   // 0b1010 (Paper/Win)
		  INVALID, // 0b1011
		  0 + 2,   // 0b1100 (Scissors/Lose)
		  3 + 3,   // 0b1101 (Scissors/Draw)
		  6 + 1,   // 0b1110 (Scissors/Win)
		  INVALID  // 0b1111 
		  );
  const __m128i scores = _mm_shuffle_epi8(table, index);
  const __m128i total_interm = _mm_hadd_epi32(scores, scores);
  const __m128i total = _mm_hadd_epi32(total_interm, total_interm);
  return _mm_cvtsi128_si32(total);
}

int main() {
  uint8_t* input = day_2_txt;
  int32_t total_score = 0;
  int32_t total_outcome = 0;
  int32_t actual[4] = {0, 0, 0, 0};
  for(int i = 0; i < day_2_txt_len; i += 16) {
    const __m128i index = create_index(_mm_loadu_si128((__m128i*)(input + i)));
    total_score += lookup_score(index);
    total_outcome += lookup_outcome(index);
  }
  printf("Total Score:   %d\nTotal Outcome: %d", total_score, total_outcome);
}
