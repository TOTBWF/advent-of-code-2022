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

./asm/day-3.s: ./src/day-3-input.c ./src/day-3.c ./asm
	clang ./src/day-3.c -march=native -O3 -S -masm=intel -I ./src/day-3-input.c -o ./asm/day-3.s

./bin/day-3: ./src/day-3-input.c ./src/day-3.c ./bin
	clang ./src/day-3.c -march=native -O3 -I ./src/day-3-input.c -o ./bin/day-3

day-3: ./bin/day-3
	./bin/day-3

## Day 4
./src/day-4-input.c:
	xxd -i ./src/day-4.txt > ./src/day-4-input.c

./asm/day-4.s: ./src/day-4-input.c ./src/day-4.c ./asm
	clang ./src/day-4.c -march=native -O3 -S -masm=intel -I ./src/day-4-input.c -o ./asm/day-4.s

./bin/day-4: ./src/day-4-input.c ./src/day-4.c ./bin
	clang ./src/day-4.c -march=native -O3 -I ./src/day-4-input.c -o ./bin/day-4

day-4: ./bin/day-4
	./bin/day-4

## Day 5
./src/day-5-input.c:
	xxd -i ./src/day-5.txt > ./src/day-5-input.c

./asm/day-5.s: ./src/day-5-input.c ./src/day-5.c ./asm
	clang ./src/day-5.c -march=native -O3 -S -masm=intel -I ./src/day-5-input.c -o ./asm/day-5.s

./bin/day-5: ./src/day-5-input.c ./src/day-5.c ./bin
	clang ./src/day-5.c -march=native -O3 -I ./src/day-5-input.c -o ./bin/day-5

day-5: ./bin/day-5
	./bin/day-5

clean:
	rm -f ./bin/* rm -f ./asm/*
