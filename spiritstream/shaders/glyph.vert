#version 330 core
layout (location = 0) in vec3 aPos;      // Base quad pos
layout (location = 1) in vec2 aTexCoord; // Base tex (0-1)

layout (location = 2) in vec3 instPos;   // Per-instance position
layout (location = 3) in vec2 instSize;  // Per-instance width/height
layout (location = 4) in vec2 instUVOffset;
layout (location = 5) in vec2 instUVSize;
layout (location = 6) in vec4 instColor;

out vec2 TexCoord;
out vec4 FragColor;

uniform vec2 scale;
uniform vec2 offset;

void main() {
    vec3 scaledPos = vec3(aPos.xy * instSize, aPos.z) + instPos;
    gl_Position = vec4(scaledPos.xy * scale + offset, scaledPos.z, 1.0);
    
    TexCoord = instUVOffset + vec2(aTexCoord.x, aTexCoord.y ) * instUVSize;
    
    FragColor = instColor;
}