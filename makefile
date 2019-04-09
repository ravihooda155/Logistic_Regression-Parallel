
CC = g++
CFLAGS = -Wall -std=c++1z
DEPS = data.h data.cpp
OBJ = data.o  lr.o
%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

logistic_regression: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

