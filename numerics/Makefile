TARGET ?= main
SRC_DIRS ?= ./src
LIB_DIRS ?= ./lib
EXT_INC_DIRS ?= /usr/local/cuda-11.3/include ./ext_inc
SYS_LIBS ?= -lncurses -lutil -lcudart -L/usr/local/cuda-11.3/lib64
CXXFLAGS+= --std=c++17 -O3

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c)
CUDASRCS := $(shell find $(SRC_DIRS) -name *.cu)
OBJS := $(addsuffix .o,$(basename $(SRCS)))
CUDAOBJS := $(addsuffix .cuo,$(basename $(CUDASRCS)))
LIBS := $(shell find $(LIB_DIRS) -name *.a)
DEPS := $(OBJS:.o=.d)


INC_DIRS = $(EXT_INC_DIRS)
INC_DIRS += $(shell find $(SRC_DIRS) -type d)


INC_FLAGS := $(addprefix -I,$(INC_DIRS))
#INC_Flags += -i/usr/local/cuda-11.3/include

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP


$(TARGET): $(OBJS) $(CUDAOBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(CUDAOBJS) $(OBJS) $(LIBS) $(SYS_LIBS) -o $@ $(LOADLIBES) $(LDLIBS)

$(CUDAOBJS): $(CUDASRCS)
	nvcc --std=c++14 -O3 -c $(INC_FLAGS) -o $@ $(addsuffix .cu, $(basename $@))


.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS) $(DEPS) $(CUDAOBJS)

#-include $(DEPS)
