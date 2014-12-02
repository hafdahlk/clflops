TARGET = clbench
CC = g++
FLAGS += -std=c++0x -O3 -Wall -pedantic
LIBS += -lOpenCL

all: $(TARGET).cpp
	$(CC) $(FLAGS) $(LIBS) $(TARGET).cpp -o $(TARGET)

clean:
	rm $(TARGET)
