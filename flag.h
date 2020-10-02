#ifndef FLAG_H
#define FLAG_H

// Βασικές Συναρτήσεις
void display();                                             // Ό,τι έχει σχέση με την απεικόνηση του μοτίβου πάει εδώ
void timer(int t);                                          // Κρατάει χρόνο για το animation
void clear();                                               // Καθαρίζει την μνήμη

// Συναρτήσεις OpenGL
void init();                                                                                            // Ξεκινά το OpenGL, αρχικοποιεί παραμετρους, κτλ
void createVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_vbo_resource);                           // Δημιουργεί ένα VBO
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_vbo_resource);                           // Διαγράφει το VBO

// Συναρτήσεις CUDA
__global__ void kernel(float4 *p, int width, int height, float time);                                   // Το kernel
void init_kernel(float4 *p, int mesh_width, int mesh_height, float time);                               // Καλεί το kernel
void run_kernel(struct cudaGraphicsResource **cuda_vbo_resource);                                       // Θέτει παραμέτρους για το kernel

#endif
