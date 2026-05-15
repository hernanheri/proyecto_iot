CXX      = g++
CXXFLAGS = -std=c++17 -O2 -fopenmp
LDFLAGS  = -lpthread

SRC = main.cpp dataset.cpp linear_regression.cpp
BIN = main

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(BIN) $(SRC) $(LDFLAGS)

clean:
	rm -f $(BIN)