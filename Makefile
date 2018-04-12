STYLE="{BasedOnStyle: llvm, IndentWidth: 8}"
FORMATTER=clang-format-6.0

all:
	gcc -g main.c -o fft -lOpenCL -lclFFT -lm

format:
	${FORMATTER} -style=${STYLE} -i main.c
	${FORMATTER} -style=${STYLE} -i pgm.h
