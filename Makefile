# Specify the targets here.
# Note each target has to have a cpp file in the "main" folder.
# TARGETS = rgtest smlpstresstest smlpcorrectnesstest
TARGETS = $(patsubst $(MAIN_DIR)/%.cpp,%,$(wildcard $(MAIN_DIR)/*.cpp))

# C++ compiler
CXX = g++
NVCC = nvcc

RT_NUM_CPUS=10

# project has an include directory
PROJECT_INCLUDE_DIR ?= ./include

# liblitmus directories
LIBLITMUS_INCLUDE_DIR ?= ../liblitmus/include
LIBLITMUS_ARCH_INCLUDE_DIR ?= ../liblitmus/arch/x86/include
LIBLITMUS_LIB_DIR ?= ../liblitmus
FEATHERTRACE_DIR ?= ../feather-trace-tools

# Set this to the base directory that contains /include/x86_65-linux-gnu and /lib/x86_64-linux-gnu
TENSORRT_BASE ?= /usr
# TensorRT directories relative to base
TENSORRT_INCLUDE_DIR ?= $(TENSORRT_BASE)/include/x86_64-linux-gnu
TENSORRT_LIB_DIR ?= $(TENSORRT_BASE)/lib/x86_64-linux-gnu

# CUDA directories
CUDA_BASE ?= /usr/local/cuda
CUDA_INCLUDE_DIR ?= $(CUDA_BASE)/include
CUDA_LIB_DIR ?= $(CUDA_BASE)/lib64

# libsmctrl directories
SMCTRL_LIB_DIR ?= ../libsmctrl
SMCTRL_INCLUDE_DIR ?= ../libsmctrl

# objects to compile
SRC_DIR = ./src
OBJ_DIR = ./obj
MAIN_DIR = ./main
# MAKE SURE MAKE SURE BIN_DIR IS JUST SOME RANDOM SUBFOLDER, BECAUSE IT GETS DELETED ON "make clean"
BIN_DIR = ./bin
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
NVCC_SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
OBJ_FILES += $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(NVCC_SRC_FILES))

# Compiler flags
CXXFLAGS = -std=c++20 -I/usr/local/include -I$(PROJECT_INCLUDE_DIR) -I$(LIBLITMUS_INCLUDE_DIR) -I$(LIBLITMUS_ARCH_INCLUDE_DIR)
CXXFLAGS += -DCONFIG_LITMUS_LOCKING_SMLP -DCONFIG_LITMUS_LOCKING_WITHARGS -DCONFIG_LITMUS_LOCKING_OMLP -DCONFIG_LITMUS_ENABLE_RELEASEGROUPS
CXXFLAGS += -DLIBLITMUS_LIB_DIR=\"$(LIBLITMUS_LIB_DIR)\" -DRT_NUM_CPUS=$(RT_NUM_CPUS) -g
CXXFLAGS += -I$(TENSORRT_INCLUDE_DIR) -I$(CUDA_INCLUDE_DIR) -I$(SMCTRL_INCLUDE_DIR)

# Linker flags
LDFLAGS =  -L$(LIBLITMUS_LIB_DIR) -llitmus
LDFLAGS += -L$(SMCTRL_LIB_DIR) -lsmctrl -L$(LIBLITMUS_LIB_DIR) -llitmus
LDFLAGS += -L$(TENSORRT_LIB_DIR) -lnvinfer -lnvinfer_plugin -lnvonnxparser -L$(CUDA_LIB_DIR) -lcudart -lcuda

all: $(TARGETS)

# Executable rule, place in bin folder
$(TARGETS): %: $(OBJ_FILES) $(OBJ_DIR)/%.o
	$(CXX) $(OBJ_DIR)/$@.o $(OBJ_FILES) -o $(BIN_DIR)/$@ $(LDFLAGS)
	@echo "Executable: $(BIN_DIR)/$@"

# Rule to compile C++ source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled: $<"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled: $<"

$(OBJ_DIR)/%.o: $(MAIN_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled: $<"

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Clean up build artifacts
clean:
	rm -f $(wildcard $(OBJ_DIR)/*.o) $(wildcard $(BIN_DIR)/*)

# Phony targets
.PHONY: all run clean