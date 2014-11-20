TARGET = clflops
CC = g++
FLAGS += -std=c++0x -O3 -Wall -pedantic
LIBS += -lOpenCL

all: hellocl.cpp
	$(CC) $(FLAGS) $(LIBS) $(TARGET).cpp -o $(TARGET)

clean: $(TARGET)
	rm $(TARGET)
