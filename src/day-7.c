#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "simd.h"
#include "day-7-input.h"

// Each column consists of a [8 * 32i] vector,
// so we divide by 8 to get the number of columns.
#define FILETREE_ROWS 175
#define FILETREE_COLUMNS (FILETREE_ROWS + 7) / 8

typedef struct filetree {
  uint32_t *adjacency;
  uint32_t *counts;
  uint32_t num_dirs;
  uint32_t cwd;
} filetree_t;

// Allocate and initialize a tree.
static inline void alloc_tree(filetree_t *tree) {
  // TODO: Make these aligned.
  tree->adjacency = calloc(FILETREE_ROWS * FILETREE_COLUMNS, sizeof(__m256i));
  tree->counts = calloc(FILETREE_COLUMNS, sizeof(__m256i));
  tree->adjacency[0] = 0xffffffff;
  tree->num_dirs = 1;
  tree->cwd = 0;
}

void print_adjacency(filetree_t *tree) {
  for(int row = 0; row < FILETREE_ROWS; row++) {
    for(int col = 0; col < FILETREE_COLUMNS*8; col++) {
      printf("%d", tree->adjacency[8*FILETREE_COLUMNS*row + col] > 0);
    }
    printf("\n");
  }
}

static inline void add_count(filetree_t *tree, uint32_t count) {
  __m256i *dir_adj = (__m256i*) (tree->adjacency + 8 * FILETREE_COLUMNS * tree->cwd);
  __m256i bdcst_count = _mm256_set1_epi32(count);

  for(int i = 0; i < FILETREE_COLUMNS; i ++) {
    __m256i masked_count = _mm256_and_si256(_mm256_loadu_si256(dir_adj + i), bdcst_count);
    __m256i *column = (__m256i*)tree->counts + i;
    _mm256_storeu_si256(column, _mm256_add_epi32(_mm256_loadu_si256(column), masked_count));
  }
}

// Create a new directory with the last set directory as the parent,
// and set the cwd to the new directory.
static inline void add_dir(filetree_t *tree) {
  __m256i *parent_adj = (__m256i*) (tree->adjacency + 8 * FILETREE_COLUMNS * tree->cwd);
  __m256i *child_adj = (__m256i*) (tree->adjacency + 8 * FILETREE_COLUMNS * tree->num_dirs);

  // Copy the parent's adjacencies to the child.
  for(int i = 0; i < FILETREE_COLUMNS; i++) {
    _mm256_storeu_si256(child_adj + i, _mm256_loadu_si256(parent_adj + i));
  }

  // Add the reflexive edge.
  tree->adjacency[8 * FILETREE_COLUMNS * tree->num_dirs + tree->num_dirs] = 0xffffffff;
  tree->cwd = tree->num_dirs;
  tree->num_dirs += 1;
}

// Change the cwd to the parent of the curent cwd.
static inline void cd_parent(filetree_t *tree) {
  __m256i *cwd_adj = (__m256i*) (tree->adjacency + 8 * FILETREE_COLUMNS * tree->cwd);
  // We need to find the right-most set bit that isn't the reflexive edge.
  for(int i = FILETREE_COLUMNS - 1; i >= 0; i--) {
    __m256i chunk = _mm256_loadu_si256(cwd_adj + i);
    // We need to mask off the reflexive edge.
    if (tree->cwd / 8 == i) {
      chunk = _mm256_andnot_si256(_mm256_expand_mask(0xf << 4 * (tree->cwd % 8)), chunk);
    }

    uint32_t lz = _lzcnt_u32(_mm256_movemask_epi8(chunk));
    // This may seem ???, but we can break it down into a series of steps.
    // First, we need to subtract 31 from the number of leading zeros to get
    // the index of the byte.
    // We also care 32 bit chunks, so we divide by 4 to get the relevant index.
    // We then need to account for which [32 * 8i] column we are in, so
    // we add on 8*i.
    uint32_t rightmost = (31 - lz) / 4 + 8 * i;
    if (lz < 32) {
      tree->cwd = rightmost;
      return;
    }
  }
}

static inline uint32_t get_counts(filetree_t *tree, uint32_t bound) {
  __m256i counts = _mm256_setzero_si256();
  for(int i = 0; i < FILETREE_COLUMNS; i++) {
    __m256i chunk = _mm256_loadu_si256((__m256i*)tree->counts + i);
    __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(bound + 1), chunk);
    counts = _mm256_add_epi32(counts, _mm256_and_si256(chunk, mask));
  }
  return _mm256_hsum_epi32(counts);
}

static inline uint32_t get_smallest(filetree_t *tree, uint32_t total, uint32_t required) {
  uint32_t unused = total - tree->counts[0];
  uint32_t bound = required - unused;
  __m256i smallest = _mm256_set1_epi32(INT32_MAX);
  for(int i = 0; i < FILETREE_COLUMNS; i++) {
    // If the value is below the required bound, clamp it to INT_MAX.
    __m256i chunk = _mm256_loadu_si256((__m256i*)tree->counts + i);
    __m256i mask = _mm256_cmpgt_epi32(chunk, _mm256_set1_epi32(bound + 1));

    chunk = _mm256_blendv_ps(_mm256_set1_epi32(INT32_MAX), chunk, mask);
    smallest = _mm256_min_epi32(smallest, chunk);
  }
  return _mm256_hmin_epi32(smallest);
}

static inline void exec_cmds(uint8_t *input, uint32_t len, filetree_t *tree) {
  // Skip the unitial 'cd /'.
  uint32_t offset = 7;
  while (offset < len) {
    __m256i chunk = _mm256_loadu_si256((__m256i*)(input + offset));
    _mm256_loadu_si256(&chunk);

    __m256i cd_mask =
      _mm256_setr_epi8('$', ' ', 'c', 'd',
		       ' ', '.', '.', ' ',
		       ' ', ' ', ' ', ' ',
		       ' ', ' ', ' ', ' ',
		       ' ', ' ', ' ', ' ',
		       ' ', ' ', ' ', ' ',
		       ' ', ' ', ' ', ' ',
		       ' ', ' ', ' ', ' ');
    __m256i cd_eq = _mm256_cmpeq_epi8(chunk, cd_mask);
    cd_eq = _mm256_andnot_si256(cd_eq, _mm256_set1_epi64x(-1));
    uint32_t cd_match = _mm_tzcnt_32(_mm256_movemask_epi8(cd_eq));
    switch (cd_match) {
    case 5:
      // We've matched the string "$ cd" but not the "..",
      // which means we are entering a directory.
      add_dir(tree);
      break;
    case 7:
      // We've matched the string "$ cd ..", so we need
      // to set the cwd to the working directory.
      cd_parent(tree);
      break;
    }

    // Now for the annoying part. If we encounter a line like "257770 thwhz.pgp",
    // we need to parse out the size, and add it to the current directory.
    // First, build up a 256 vector that will be our digit mask.
    __m128i chunk_lo = _mm256_castsi256_si128(chunk);
    __m128i is_digit_mask = _mm_cmpgt_epi8(chunk_lo, _mm_set1_epi8('0' - 1));
    is_digit_mask = _mm_and_si128(is_digit_mask, _mm_cmplt_epi8(chunk_lo, _mm_set1_epi8('9' + 1)));
    is_digit_mask = _mm_andnot_si128(is_digit_mask, _mm_set1_epi64x(-1));
    uint32_t num_digits = _mm_tzcnt_32(_mm_movemask_epi8(is_digit_mask));
    switch (num_digits) {
    case 4:
      add_count(tree, parse_4_digits(chunk_lo));
      break;
    case 5:
      add_count(tree, parse_5_digits(chunk_lo));
      break;
    case 6:
      add_count(tree, parse_6_digits(chunk_lo));
      break;
    }

    __m256i newline_eq = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\n'));
    offset += _mm_tzcnt_32(_mm256_movemask_epi8(newline_eq)) + 1;
  }
}

int main() {
  filetree_t tree;
  printf("==========[Preprocess]======\n");
  clock_t preprocess_start = clock();
  alloc_tree(&tree);
  exec_cmds(input_day_7_txt, input_day_7_txt_len, &tree);
  double preprocess_time = (double)(clock() - preprocess_start) / CLOCKS_PER_SEC;
  printf("Completed in %1.0lf microseconds\n", preprocess_time * pow(10, 6));
  printf("==========[Part 1]==========\n");
  clock_t part_1_start = clock();
  uint32_t counts = get_counts(&tree, 100000);
  double part_1_time = (double)(clock() - part_1_start) / CLOCKS_PER_SEC;
  printf("Part 1: %d\n", counts);
  printf("Completed in %1.0lf microseconds\n", part_1_time * pow(10, 6));
  printf("==========[Part 2]==========\n");
  clock_t part_2_start = clock();
  uint32_t smallest = get_smallest(&tree, 70000000, 30000000);
  double part_2_time = (double)(clock() - part_2_start) / CLOCKS_PER_SEC;
  printf("Part 2: %d\n", smallest);
  printf("Completed in %1.0lf microseconds\n", part_2_time * pow(10, 6));
}
