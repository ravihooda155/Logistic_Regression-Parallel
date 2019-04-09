
CC = g++
CFLAGS = -Wall -std=c++1z
DEPS = utility.h main.cpp
OBJ = utility.o  lr.o
%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

logistic_regression: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

