#ifndef FLAG_H
#define FLAG_H

void display();
void timer(int t);
void clear();

void init();                                                                                            
void createVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_vbo_resource);                           
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_vbo_resource);


__global__ void kernel(float4 *p, int width, int height, float time);                                   
void init_kernel(float4 *p, int mesh_width, int mesh_height, float time);                               
void run_kernel(struct cudaGraphicsResource **cuda_vbo_resource);                                       

#endif
