STYLE="{BasedOnStyle: llvm, IndentWidth: 8}"
FORMATTER=clang-format-6.0

CFLAGS = -march=native -O3 -ffast-math -fprefetch-loop-arrays -ftree-vectorize
#CFLAGS += -g

all:
	gcc $(CFLAGS) main.c -o fft -lOpenCL -lclFFT -lm

format:
	${FORMATTER} -style=${STYLE} -i main.c
	${FORMATTER} -style=${STYLE} -i pgm.h

clean:
	rm -f fft
