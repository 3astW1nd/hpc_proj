# Use nvcc and optimize
CC      := nvcc
CFLAGS  := -O2 -lcublas -arch=sm_75

EXE     := nn.exe
SRC     := nn.cu

all: $(EXE) run

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)
