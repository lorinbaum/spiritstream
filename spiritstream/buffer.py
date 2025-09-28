from spiritstream.bindings.opengl import *
from spiritstream.shader import Shader

def glUnbindBuffers():
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

class BaseQuad:
    """base unit quad used with instancing"""
    def __init__(self):
        #                top-left                 # top-right              # bottom-left            # bottom-right
        self.vertices = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        self.indices = [0, 1, 2, 1, 2, 3]
        self.VBO, self.VAO, self.EBO = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        glGenVertexArrays(1, ctypes.byref(self.VAO))
        glGenBuffers(1, ctypes.byref(self.VBO))
        glGenBuffers(1, ctypes.byref(self.EBO))
        
        glBindVertexArray(self.VAO) # must be bound before EBO
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)

        quad_vertices_ctypes = (ctypes.c_float * len(self.vertices))(*self.vertices)
        quad_indices_ctypes = (ctypes.c_uint * len(self.indices)) (*self.indices)
        # pre allocate buffers
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(quad_vertices_ctypes), quad_vertices_ctypes, GL_STATIC_DRAW)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(quad_indices_ctypes), quad_indices_ctypes, GL_STATIC_DRAW)

        glUnbindBuffers()
        self.deleted = False # multiple delete calls since this is base for other buffers

    def delete(self):
        """Free OpenGL resources. Call before program exit"""
        glDeleteBuffers(1, self.VBO)
        glDeleteBuffers(1, self.EBO)
        glDeleteVertexArrays(1, self.VAO)

class Buffer:
    def __init__(self, *args, **kwargs): raise NotImplementedError
    
    def add(self, data):
        assert len(data) % self.stride == 0
        self.data.extend(data)
        self.count += len(data) // self.stride
        self.changed = True

    def replace(self, data):
        assert len(data) % self.stride == 0
        self.data = data
        self.count = len(data) // self.stride
        self.changed = True

    def clear(self):
        self.data.clear()
        self.count = 0

    def draw(self, scale, offset):
        if hasattr(self, "texture"):
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glDisable(GL_DEPTH_TEST)
            glDepthMask(GL_FALSE)  # Prevent writing to the depth buffer
        if self.changed: # upload buffer. NOTE: no mechanism shrinks the buffer if fewer quads are needed as before. same applies to quad buffer
            data_c = (ctypes.c_float * (length:=self.count*self.stride))(*self.data[:length])
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
            glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(data_c), data_c, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.changed = False
        self.shader.use()
        self.shader.setUniform("scale", scale, "2f") # inverted y axis. from my view (0,0) is the top left corner, like in browsers
        self.shader.setUniform("offset", offset, "2f")
        glBindVertexArray(self.VAO)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, self.count)
        if hasattr(self, "texture"):
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)
        glBindVertexArray(0)
    
    def delete(self):
        """Free OpenGL resources. Call before program exit"""
        glDeleteBuffers(1, self.VBO)
        glDeleteVertexArrays(1, self.VAO)

class TexQuadBuffer(Buffer):
    """Buffer for textured quad instances like glyphs and images"""
    def __init__(self, base:BaseQuad, texture, shader:Shader, tinted_atlas:bool=False):
        """Create instance data buffer, vertex attribute array and instance variables.
        tinted_atlas is for glyphs, where instance data is expected to include uv offset and uv size as well as a color for tinting"""
        self.base, self.shader, self.changed, self.count, self.stride = base, shader, False, 0, 5 + (8 if tinted_atlas else 0)
        self.texture = texture
        self.data = [] # x, y, z, size w, h, uv offset x, y, uv size w, h, color r, g, b, a

        stride_c = self.stride * ctypes.sizeof(ctypes.c_float)

        self.VAO = ctypes.c_uint()
        glGenVertexArrays(1, ctypes.byref(self.VAO))
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.base.VBO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.base.EBO)
        
        # base quad data attributes: vertex position (loc 0), texture position (loc 1)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(1)

        self.VBO = ctypes.c_uint() # instance buffer
        glGenBuffers(1, ctypes.byref(self.VBO))
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        
        # Instance attributes (locations 2+; divisor=1 for per-instance)
        # pos (loc 2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        # size (loc 3)
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)

        if tinted_atlas:
          # uv offset (loc 4)
          glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(5*ctypes.sizeof(ctypes.c_float)))
          glEnableVertexAttribArray(4)
          glVertexAttribDivisor(4, 1)
          # uv size (loc 5)
          glVertexAttribPointer(5, 2, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(7*ctypes.sizeof(ctypes.c_float)))
          glEnableVertexAttribArray(5)
          glVertexAttribDivisor(5, 1)
          # color (loc 6)
          glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(9*ctypes.sizeof(ctypes.c_float)))
          glEnableVertexAttribArray(6)
          glVertexAttribDivisor(6, 1)

        glUnbindBuffers()

class QuadBuffer(Buffer):
    """Buffer for untextured quad instances like backgrounds, underlines and images"""
    def __init__(self, base, shader):
        """Create instance data buffer, vertex attribute array and instance variables"""
        self.base, self.shader, self.changed, self.count, self.stride = base, shader, False, 0, 9
        self.data = [] # x, y, z, size w, h, color r, g, b, a

        stride_c = self.stride * ctypes.sizeof(ctypes.c_float)

        self.VAO = ctypes.c_uint()
        glGenVertexArrays(1, ctypes.byref(self.VAO))
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.base.VBO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.base.EBO)
        
        # base quad data attributes: vertex position (loc 0), texture position (loc 1)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(1)

        self.VBO = ctypes.c_uint() # instance buffer
        glGenBuffers(1, ctypes.byref(self.VBO))
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)

        # Instanced attributes (locations 2+; divisor=1 for per-instance)
        # pos (loc 2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        # size (loc 3)
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        # color (loc 4)
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(5*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)

        glUnbindBuffers()