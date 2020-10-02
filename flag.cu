#include <cstdio>
#include <cstdlib>
#include <cmath>
// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
// OpenGL
#include <GL/freeglut.h>
#include <helper_gl.h>
// Custom Header
#include "flag.h"

// Constant declarations
const int window_width = 1600;
const int window_height = 900;
const int mesh_width = 64;
const int mesh_height = 64;
const int REFRESH_DELAY = 10;

// Αrgument declations
int argc = 1;
char *argv[1] = {};

// Θέση, περιστροφή, παράμετροι του animation
float animation_freq = 0;
float rotate_x = 50;
float rotate_y = 180;
float rotate_z = 25;
float translate_z = -2.5;
float translate_x = 0.5;
float translate_y = -0.5;

// VBO
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *vbo_buffer = 0;

// 
////////// CUDA KERNEL FUNCTIONS //////////
// 
__global__ void kernel(float4 *p, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Συντεταγμένες
    float w = x / (float)width;
    float h = y / (float)height * 0.7;
    
    // Μοτίβο
    float frequency = 2.5;
    //float wave = sinf(w * frequency + time - 10000000) * cosf(sqrt(w) + 10); //Αυτό είναι ένα τυχαίο μοτίβο που μου έκανε εντύπωση
    // Το κύμα που θα ακολουθηθεί
    float wave = sinf((w - 10) * frequency + time) * cosf(sqrt(w) + 10) + h;
    // Εξαγωγή
    p[y * width + x] = make_float4(w, wave, h, 1);
}

void init_kernel(float4 *p, int mesh_width, int mesh_height, float time) {
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    kernel <<< grid, block >>>(p, mesh_width, mesh_height, time);
}

void run_kernel(struct cudaGraphicsResource **cuda_vbo_resource) {
    // map OpenGL buffer
    float4 *ptr;
    cudaGraphicsMapResources(1, cuda_vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes, *cuda_vbo_resource);
    
    // initialize kernel
    init_kernel(ptr, mesh_width, mesh_height, animation_freq);
    
    // unmap buffer
    cudaGraphicsUnmapResources(1, cuda_vbo_resource, 0);
}
 // 
 ////////// OPENGL FUNCTIONS //////////
 // 
void init() {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Ergasia PAS - Flag");
    glutDisplayFunc(display);
    glutTimerFunc(REFRESH_DELAY, timer, 0);
    
    // Αρχικοποιήσεις
    glClearColor(0, 0, 0, 1);
    glDisable(GL_DEPTH_TEST);
    
    // Viewport
    glViewport(0, 0, window_width, window_height);
    
    // Προβολή
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10);
}
void createVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_vbo_resource) {
    // Δημιουργούμε το buffer
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    
    // Αρχικοποιούμε το buffer
    size_t size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    
    // Εγγράφουμε το buffer στην συσκευή
    cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, *vbo, cudaGraphicsMapFlagsWriteDiscard);
}
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_vbo_resource) {
    // Καταργούμε την εγγραφή του buffer
    cudaGraphicsUnregisterResource(*cuda_vbo_resource);
    
    // Διαγράφουμε το buffer και μηδενίζουμε τον δείκτη
    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);
    *vbo = 0;
}
// 
////////// MAIN FUNCTIONS //////////
// 
void display() {
    // Καλουμε την συνάρτηση που καλεί το kernel
    run_kernel(&cuda_vbo_resource);
    
    // Καθαρίζουμε την οθόνη
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Θέτουμε τις ιδιότητες του μοντέλου
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translate_x, translate_y, translate_z);
    glRotatef(rotate_x, 1, 0, 0);
    glRotatef(rotate_y, 0, 1, 0);
    glRotatef(rotate_z, 0, 0, 1);
    
    // Rendering
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glColor4f(0, 0.5, 1, 0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);
    
    glutSwapBuffers();
    
    // Η συχνότητα του animation
    animation_freq += 0.03;
}

void timer(int t) {
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timer, 0);
    }
}

void clear() {
    if (vbo) deleteVBO(&vbo, &cuda_vbo_resource);
}

int main() {
    size_t N = mesh_width * mesh_height * sizeof(float);
    void *imageData = malloc(N);
    setenv("DISPLAY", ":0", 0);
    init();
    glutCloseFunc(clear);
    createVBO(&vbo, &cuda_vbo_resource);
    run_kernel(&cuda_vbo_resource);
    glutMainLoop();
    cudaMalloc((void **)&vbo_buffer, 4 * N);
    cudaDeviceSynchronize();
    cudaMemcpy(imageData, &vbo_buffer, N, cudaMemcpyDeviceToHost);
    cudaFree(vbo_buffer);
    vbo_buffer = 0;
    return 0;
}
