.PHONY: day-1 day-2 day-3 clean

./bin:
	mkdir ./bin

./asm:
	mkdir ./asm

## Day 1
./src/day-1-input.c:
	xxd -i day-1.txt > day-1-input.c

./bin/day-1: ./src/day-1-input.c ./src/day-1.c ./bin
	clang ./src/day-1.c -march=native -O3 -I ./src/day-1-input.c -o ./bin/day-1

./asm/day-1.s: ./src/day-1-input.c ./src/day-1.c ./asm
	clang ./src/day-1.c -march=native -O3 -S -masm=intel -I ./src/day-1-input.c -o ./asm/day-1.s

day-1: ./bin/day-1
	./bin/day-1

## Day 2
./src/day-2-input.c:
	xxd -i day-2.txt > day-2-input.c

./asm/day-2.s: ./src/day-2-input.c ./src/day-2.c ./asm
	clang ./src/day-2.c -march=native -O3 -S -masm=intel -I ./src/day-2-input.c -o ./asm/day-2.s

./bin/day-2: ./src/day-2-input.c ./src/day-2.c ./bin
	clang ./src/day-2.c -march=native -O3 -I ./src/day-2-input.c -o ./bin/day-2

day-2: ./bin/day-2
	./bin/day-2

## Day 3
./src/day-3-input.c:
	xxd -i ./src/day-3.txt > ./src/day-3-input.c

./bin/day-3: ./src/day-3-input.c ./src/day-3.c ./bin
	clang ./src/day-3.c -march=native -O3 -I ./src/day-3-input.c -o ./bin/day-3

day-3: ./bin/day-3
	./bin/day-3

clean:
	rm -f ./bin/* rm -f ./asm/*
