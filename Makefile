.PHONY: all run clean
.PRECIOUS: ./input/%-input.h

all: ./bin/day-1 ./bin/day-2 ./bin/day-3 ./bin/day-4 ./bin/day-5 ./bin/day-6

run: ./bin/day-$(day)
	./bin/day-$(day)

clean:
	rm -f ./bin/* rm -f ./asm/*

./input/%-input.h: ./input/%.txt
	xxd -i $< > $@

./asm/%.s: ./src/%.c ./input/%-input.h ./asm
	clang $< -march=native -O3 -S -I ./input/ -masm=intel -o $@

./bin/%: ./src/%.c ./input/%-input.h ./src/simd.h ./bin
	clang $< -march=native -O3 -I ./input/ -o $@

./bin:
	mkdir ./bin

./asm:
	mkdir ./asm
