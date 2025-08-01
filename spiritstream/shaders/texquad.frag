#version 330 core
in vec2 TexCoord;
in vec4 FragColor;
out vec4 color;

uniform sampler2D glyphAtlas;

void main() {
    float alpha = texture(glyphAtlas, TexCoord).r;
    color = vec4(FragColor.rgb, alpha * FragColor.a);
}
