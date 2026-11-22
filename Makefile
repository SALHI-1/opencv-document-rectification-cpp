# ============================================================================
# Makefile - Redressement de Perspective de Documents (OpenCV C++)
# Auteur : Mohammed SALHI
# ============================================================================

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

TARGET = perspective_transform
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET) *_redresse*

.PHONY: all clean
