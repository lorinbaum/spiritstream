# Compiler and flags
CC = gcc
CFLAGS = -fPIC -O2 -Wall
LDFLAGS = -shared
BUILD_DIR = spiritstream/bindings/build

# Paths
GLFW_DIR = spiritstream/bindings/lib/glfw-3.4
NANOJPEG_SRC = spiritstream/bindings/lib/nanojpeg.c
LIBSPNG_SRC = spiritstream/bindings/lib/libspng-0.7.4/spng.c

all: $(BUILD_DIR)/libglfw.so $(BUILD_DIR)/libnanojpeg.so $(BUILD_DIR)/libspng.so

$(BUILD_DIR)/libglfw.so:
	mkdir -p $(BUILD_DIR)
	cd $(GLFW_DIR) && cmake -S . -B ../../build/glfw-3.4 -D BUILD_SHARED_LIBS=ON -D GLFW_BUILD_X11=1 -D GLFW_BUILD_WAYLAND=0 -D GLFW_BUILD_DOCS=0 -D GLFW_BUILD_EXAMPLES=0 -D GLFW_BUILD_TESTS=0
	cd $(BUILD_DIR)/glfw-3.4 && make
	cp $(BUILD_DIR)/glfw-3.4/src/libglfw.so $(BUILD_DIR)/libglfw.so
	rm -rf $(BUILD_DIR)/glfw-3.4

$(BUILD_DIR)/libnanojpeg.so:
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $(NANOJPEG_SRC) -o $@

$(BUILD_DIR)/libspng.so:
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $(LIBSPNG_SRC) -o $@ -lz

clean:
	rm -rf $(BUILD_DIR)/*
