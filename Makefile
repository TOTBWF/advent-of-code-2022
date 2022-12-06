.PHONY: run clean
.PRECIOUS: ./input/%-input.h

./bin:
	mkdir ./bin

./asm:
	mkdir ./asm

./input/%-input.h: ./input/%.txt
	xxd -i $< > $@

./asm/%.s: ./src/%.c ./input/%-input.h ./asm
	clang $< -march=native -O3 -S -I ./input/ -masm=intel -o $@

./bin/%: ./src/%.c ./input/%-input.h  ./bin
	clang $< -march=native -O3 -I ./input/ -o $@

run: ./bin/day-$(day)
	./bin/day-$(day)

clean:
	rm -f ./bin/* rm -f ./asm/*
