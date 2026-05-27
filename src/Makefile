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
	
mpi: linear_regression_mpi.cpp dataset.cpp
	mpic++ -std=c++17 -O2 -o main_mpi \
		linear_regression_mpi.cpp dataset.cpp

run_mpi: mpi
	mpirun -np 4 ./main_mpi

lib:
	$(CXX) $(CXXFLAGS) -shared -fPIC -o iot_lib.so \
		iot_lib.cpp dataset.cpp linear_regression.cpp $(LDFLAGS)