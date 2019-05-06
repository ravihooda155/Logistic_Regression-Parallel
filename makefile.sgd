
CC = g++
CFLAGS = -Wall -std=c++11 -fopenmp
DEPS = utility.h logistic.h 
OBJ = utility.o logistic.o sgd.o 
%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

logistic_regression_sgd: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

