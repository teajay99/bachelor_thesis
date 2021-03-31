TARGET ?= main
SRC_DIRS ?= ./src
LIB_DIRS ?= ./lib
EXT_INC_DIRS ?= ./ext_inc
SYS_LIBS ?= -lncurses -lutil -L/usr/local/cuda/lib64 -lcuda -lcudart
CXXFLAGS+= --std=c++17 -O3

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c)
CUDASRCS := $(shell find $(SRC_DIRS) -name *.cu)
OBJS := $(addsuffix .o,$(basename $(SRCS)))
CUDAOBJS := $(addsuffix .cuo,$(basename $(CUDASRCS)))
LIBS := $(shell find $(LIB_DIRS) -name *.a)
DEPS := $(OBJS:.o=.d)


INC_DIRS += $(shell find $(EXT_INC_DIRS) -maxdepth 1 -type d)
INC_DIRS += $(shell find $(SRC_DIRS) -type d)


INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP


$(TARGET): $(OBJS) $(CUDAOBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(CUDAOBJS) $(OBJS) $(LIBS) $(SYS_LIBS) -o $@ $(LOADLIBES) $(LDLIBS)

$(CUDAOBJS): $(CUDASRCS)
	nvcc --std=c++14 -c $(INC_FLAGS) -o $@ $(addsuffix .cu, $(basename $@))


.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS) $(DEPS) $(CUDAOBJS)

#-include $(DEPS)
