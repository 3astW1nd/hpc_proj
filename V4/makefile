CC = nvcc
CFLAGS = -O2

EXE = nn.exe
SRC = nn.cu

all: $(EXE) run

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) 

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)
