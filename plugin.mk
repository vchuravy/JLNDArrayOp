JGPU_SRC = $(wildcard plugin/jgpu/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(JGPU_SRC))
