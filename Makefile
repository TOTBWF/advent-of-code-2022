.PHONY: day-1 day-2

./src/day-1-input.c:
	xxd -i day-1.txt > day-1-input.c

./bin/day-1: ./src/day-1-input.c
	clang ./src/day-1.c -march=native -O2 -I ./src/day-1-input.c -o ./bin/day-1

./src/day-2-input.c:
	xxd -i day-2.txt > day-2-input.c

./bin/day-2: ./src/day-2-input.c
	clang ./src/day-2.c -march=native -O2 -I ./src/day-2-input.c -o ./bin/day-2

day-1: ./bin/day-1
	./bin/day-1

day-2: ./bin/day-2
	./bin/day-2
