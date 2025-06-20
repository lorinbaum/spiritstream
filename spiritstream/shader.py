from spiritstream.bindings.opengl import *
from typing import Union, List

class Shader:
    def __init__(self, vertex_shader_source:str, fragment_shader_source:str, uniforms:List[str]):
        vertex_shader = self._compile(GL_VERTEX_SHADER, vertex_shader_source)
        fragment_shader = self._compile(GL_FRAGMENT_SHADER, fragment_shader_source)
        self.program = glCreateProgram()
        glAttachShader(self.program, vertex_shader)
        glAttachShader(self.program, fragment_shader)
        glLinkProgram(self.program)
        success = ctypes.c_int(0)
        glGetProgramiv(self.program, GL_LINK_STATUS, ctypes.byref(success))
        if not success.value:
            error_log = ctypes.create_string_buffer(512)
            glGetProgramInfoLog(self.program, 512, None, error_log)
            raise RuntimeError(f"Program link failed: {error_log.value.decode()}")
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        self.uniforms_locations = {u:glGetUniformLocation(self.program, u.encode()) for u in uniforms}

    def use(self): glUseProgram(self.program)

    def setUniform(self, name:str, value:Union[int, tuple], dtype:str):
        self.use()
        assert dtype in ["1i", "2f", "3f"]
        assert name in self.uniforms_locations
        match dtype:
            case "1i":
                assert isinstance(value, int)
                glUniform1i(self.uniforms_locations[name], value)
            case "2f":
                assert isinstance(value, tuple) and len(value) == 2
                glUniform2f(self.uniforms_locations[name], *value)
            case "3f":
                assert isinstance(value, tuple) and len(value) == 3
                glUniform3f(self.uniforms_locations[name], *value)

    def delete(self): glDeleteProgram(self.program)
        
    def _compile(self, shader_type, source_code):
        shader_id = glCreateShader(shader_type)
        src_encoded = source_code.encode('utf-8')
        
        src_ptr = ctypes.c_char_p(src_encoded)
        src_array = (ctypes.c_char_p * 1)(src_ptr)
        length_array = (ctypes.c_int * 1)(len(src_encoded))
        
        glShaderSource(shader_id, 1, src_array, length_array)
        glCompileShader(shader_id)
        
        success = ctypes.c_int(0)
        glGetShaderiv(shader_id, GL_COMPILE_STATUS, ctypes.byref(success))
        if not success.value:
            error_log = ctypes.create_string_buffer(512)
            glGetShaderInfoLog(shader_id, 512, None, error_log)
            raise RuntimeError(f"Shader compilation failed: {error_log.value.decode()}")
        return shader_id