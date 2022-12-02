## SIMD Advent of Code 2022

This repo contains my solutions for [The Advent of Code 2022](https://adventofcode.com/2022).
So far I've been doing most of my solutions in C with heavy use of SIMD.

## Compiling

To run the solution for the nth day, use `make day-<n>`.

## Notes

I've been using `xxd -i` to embed the input files into the binaries;
this avoids having to write annoying IO code, and also radically speeds
up the solutions; so far everything has been on the order of microseconds.
