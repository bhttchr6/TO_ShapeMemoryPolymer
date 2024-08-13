include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
SRCDIR := src
BUILDDIR := build
TARGETDIR := bin
DEPS := analysis.h MMA.h

TARGET := test # executable

CC     =g++
INC    = -I${HOME}/eigen3 -I${PETSC_DIR}/include -I${PETSC_DIR}/arch-linux2-c-debug/include
SRCS   = $(shell find $(SRCDIR) -type f -name *.cpp)
OBJ    = $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SRCS:.cpp=.o))
CLEANFILES =${OBJ} test




$(TARGET): $(OBJ)  chkopts
	-${CLINKER}   -o $(TARGET) ${OBJ} ${PETSC_SYS_LIB}

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp 
	-${CLINKER}  ${PETSC_SYS_LIB} ${INC} -c $< -o $@ 



#main.o: main.cpp
#	-${CLINKER}  ${PETSC_SYS_LIB} ${INC} -c main.cpp -o main.o 

#all: test
#CFLAGS     = -I${PETSC_DIR}/include
#FFLAGS     = -I${PETSC_DIR}/include/finclude
#SOURCESC   = main.cpp
#SOURCESF   =
#OBJ        =$(SOURCESC:.cpp=.o)
#CLEANFILES =${OBJ} test

#test: ${OBJ} chkopts
#	-${CLINKER} -o test ${OBJ} ${PETSC_SYS_LIB}
