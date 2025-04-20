#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// CUDA error checking
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Timer function
float get_time(clock_t start) {
    return (float)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
float** allocateMatrix(int rows, int cols) {
    float** mat = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (float*)malloc(cols * sizeof(float));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(float** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Neural network structure
typedef struct {
    float** W1;
    float** W2;
    float* b1;
    float* b2;
} NeuralNetwork;

// GPU network structure
typedef struct {
    float* W1;  // Flattened matrix
    float* W2;  // Flattened matrix
    float* b1;
    float* b2;
} NeuralNetworkGPU;

// ReLU activation for GPU
__global__ void relu_gpu(float* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        x[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

// Softmax activation for GPU
__global__ void softmax_gpu(float* output) {
    float maxVal = output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) maxVal = fmaxf(output[i], maxVal);
    float sum = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float e = expf(output[i] - maxVal);
        output[i] = e;
        sum += e;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] /= sum;
    }
}

// Matrix multiplication for GPU
__global__ void matrixMulGpu(const float* __restrict__ W,
    const float* __restrict__ input,
    float* res,
    const float* __restrict__ b,
    int size1,
    int size2) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < size1) {
float sum = __ldg(&b[i]);
for (int j = 0; j < size2; j++) {
sum += __ldg(&W[i * size2 + j]) * __ldg(&input[j]);
}
res[i] = sum;
}
}

// Output layer gradient computation for GPU
__global__ void outputLayerGradient_gpu(float* d_output,
    const float* output,
    const float* target) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < OUTPUT_SIZE) {
d_output[i] = output[i] - target[i];
}
}

// Hidden layer gradient computation for GPU
__global__ void hiddenLayerGradient_gpu(float* d_hidden,
    const float* W2,
    const float* d_output,
    const float* hidden) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < HIDDEN_SIZE) {
float sum = 0.0f;
for (int j = 0; j < OUTPUT_SIZE; j++) {
sum += __ldg(&W2[j * HIDDEN_SIZE + i]) * d_output[j];
}
d_hidden[i] = sum * (hidden[i] > 0.0f);
}
}

// Weight gradient descent for GPU
__global__ void gradientDescentW_gpu(float* W,
    const float* d1,
    const float* arr,
    int size1,
    int size2,
    float learning_rate) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
if (i < size1 && j < size2) {
W[i * size2 + j] -= learning_rate * d1[i] * arr[j];
}
}

// Bias gradient descent for GPU
__global__ void gradientDescentB_gpu(float* B,
    const float* d1,
    int size,
    float learning_rate) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < size) {
B[i] -= learning_rate * d1[i];
}
}

// 2D to 1D array flattening
float* flattenMatrix(float** mat, int rows, int cols) {
    float* flat = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            flat[i * cols + j] = mat[i][j];
    return flat;
}

// 1D to 2D array unflattening
void unflattenMatrix(float* flat, float** mat, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mat[i][j] = flat[i * cols + j];
}


// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    net->b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((float)rand() / RAND_MAX) * 0.01f;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((float)rand() / RAND_MAX) * 0.01f;

    return net;
}

// Initialize GPU network from CPU network
NeuralNetworkGPU* createNetworkGPU(NeuralNetwork* cpuNet) {
    NeuralNetworkGPU* gpuNet = (NeuralNetworkGPU*)malloc(sizeof(NeuralNetworkGPU));
    float* flat_W1 = flattenMatrix(cpuNet->W1, HIDDEN_SIZE, INPUT_SIZE);
    float* flat_W2 = flattenMatrix(cpuNet->W2, OUTPUT_SIZE, HIDDEN_SIZE);

    CHECK_CUDA_ERROR(cudaMalloc(&gpuNet->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpuNet->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpuNet->b1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpuNet->b2, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->W1, flat_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->W2, flat_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->b1, cpuNet->b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->b2, cpuNet->b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    free(flat_W1);
    free(flat_W2);
    return gpuNet;
}

// Free GPU network
void freeNetworkGPU(NeuralNetworkGPU* net) {
    CHECK_CUDA_ERROR(cudaFree(net->W1));
    CHECK_CUDA_ERROR(cudaFree(net->W2));
    CHECK_CUDA_ERROR(cudaFree(net->b1));
    CHECK_CUDA_ERROR(cudaFree(net->b2));
    free(net);
}

// Copy network from GPU to CPU
void copyNetworkToCPU(NeuralNetworkGPU* gpuNet, NeuralNetwork* cpuNet) {
    float* flat_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* flat_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(flat_W1, gpuNet->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(flat_W2, gpuNet->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(cpuNet->b1, gpuNet->b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(cpuNet->b2, gpuNet->b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    unflattenMatrix(flat_W1, cpuNet->W1, HIDDEN_SIZE, INPUT_SIZE);
    unflattenMatrix(flat_W2, cpuNet->W2, OUTPUT_SIZE, HIDDEN_SIZE);
    free(flat_W1);
    free(flat_W2);
}

// Forward pass on GPU with stream parameter
void forwardGPU(NeuralNetworkGPU* net,
                float* d_input,
                float* d_hidden,
                float* d_output,
                cudaStream_t stream) {
    int blockSize = 256;
    int hiddenGrid = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    int outputGrid = (OUTPUT_SIZE + blockSize - 1) / blockSize;

    matrixMulGpu<<<hiddenGrid, blockSize, 0, stream>>>(net->W1, d_input, d_hidden, net->b1, HIDDEN_SIZE, INPUT_SIZE);
    relu_gpu<<<hiddenGrid, blockSize, 0, stream>>>(d_hidden, HIDDEN_SIZE);
    matrixMulGpu<<<outputGrid, blockSize, 0, stream>>>(net->W2, d_hidden, d_output, net->b2, OUTPUT_SIZE, HIDDEN_SIZE);
    softmax_gpu<<<1, 1, 0, stream>>>(d_output);
}

// Backward pass on GPU with stream parameter
void backwardGPU(NeuralNetworkGPU* net,
                 float* d_input,
                 float* d_hidden,
                 float* d_output,
                 float* d_target,
                 cudaStream_t stream) {
    float *d_d_output, *d_d_hidden;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_output, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_hidden, HIDDEN_SIZE * sizeof(float)));

    int blockSize = 256;
    int outputGrid = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    int hiddenGrid = (HIDDEN_SIZE + blockSize - 1) / blockSize;

    outputLayerGradient_gpu<<<outputGrid, blockSize, 0, stream>>>(d_d_output, d_output, d_target);
    hiddenLayerGradient_gpu<<<hiddenGrid, blockSize, 0, stream>>>(d_d_hidden, net->W2, d_d_output, d_hidden);

    dim3 outBlock(16,16);
    dim3 outGrid((OUTPUT_SIZE+outBlock.x-1)/outBlock.x, (HIDDEN_SIZE+outBlock.y-1)/outBlock.y);
    dim3 hidBlock(16,16);
    dim3 hidGrid((HIDDEN_SIZE+hidBlock.x-1)/hidBlock.x, (INPUT_SIZE+hidBlock.y-1)/hidBlock.y);

    gradientDescentW_gpu<<<outGrid, outBlock, 0, stream>>>(net->W2, d_d_output, d_hidden, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    gradientDescentB_gpu<<<outputGrid, blockSize, 0, stream>>>(net->b2, d_d_output, OUTPUT_SIZE, LEARNING_RATE);
    gradientDescentW_gpu<<<hidGrid, hidBlock, 0, stream>>>(net->W1, d_d_hidden, d_input, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    gradientDescentB_gpu<<<hiddenGrid, blockSize, 0, stream>>>(net->b1, d_d_hidden, HIDDEN_SIZE, LEARNING_RATE);

    CHECK_CUDA_ERROR(cudaFree(d_d_output));
    CHECK_CUDA_ERROR(cudaFree(d_d_hidden));
}

// Train network with communication overlap and streams
void train(NeuralNetwork* net, float** images, float** labels, int numImages) {
    clock_t total_start = clock();
    NeuralNetworkGPU* gpuNet = createNetworkGPU(net);

    float *d_input, *d_hidden, *d_output, *d_target;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(float)));

    float* h_output;
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_output, OUTPUT_SIZE * sizeof(float), cudaHostAllocDefault));

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    for (int epoch=0; epoch<EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float loss = 0.0f;
        int correct = 0;
        for (int i=0; i<numImages; i++) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, images[i], INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_target, labels[i], OUTPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream));

            forwardGPU(gpuNet, d_input, d_hidden, d_output, stream);
            backwardGPU(gpuNet, d_input, d_hidden, d_output, d_target, stream);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));

            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

            for (int k=0; k<OUTPUT_SIZE; k++) {
                loss -= labels[i][k] * logf(fmaxf(h_output[k], 1e-10f));
            }
            int pred=0, act=0;
            for (int j=1; j<OUTPUT_SIZE; j++) {
                if (h_output[j]>h_output[pred]) pred=j;
                if (labels[i][j]>labels[i][act]) act=j;
            }
            if (pred==act) correct++;
        }
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch+1, loss/numImages, correct/(float)numImages*100.0f, get_time(epoch_start));
    }

    printf("Total training time: %.3fs\n", get_time(total_start));

    copyNetworkToCPU(gpuNet, net);
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    freeNetworkGPU(gpuNet);

    CHECK_CUDA_ERROR(cudaFreeHost(h_output));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_target));
}

// Evaluate accuracy on test data (unchanged logic)
void evaluate(NeuralNetwork* net, float** images, float** labels, int numImages) {
    NeuralNetworkGPU* gpuNet = createNetworkGPU(net);
    float *d_input, *d_hidden, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, INPUT_SIZE*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hidden, HIDDEN_SIZE*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, OUTPUT_SIZE*sizeof(float)));
    float* h_output;
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_output, OUTPUT_SIZE*sizeof(float), cudaHostAllocDefault));
    cudaStream_t stream; CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    int correct=0;
    for (int i=0; i<numImages; i++) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, images[i], INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream));
        forwardGPU(gpuNet, d_input, d_hidden, d_output, stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        int pred=0, act=0;
        for (int j=1; j<OUTPUT_SIZE; j++) {
            if (h_output[j]>h_output[pred]) pred=j;
            if (labels[i][j]>labels[i][act]) act=j;
        }
        if (pred==act) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", correct/(float)numImages*100.0f);
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    freeNetworkGPU(gpuNet);
    CHECK_CUDA_ERROR(cudaFreeHost(h_output));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

// Read MNIST dataset
float** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    float** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

float** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    float** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// Free network memory
void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network - Further Optimization (V3)\n\n");
    
    // Load MNIST dataset
    float** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    float** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    float** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    float** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);
    
    // Create and initialize network
    NeuralNetwork* net = createNetwork();
    
    // Train network
    train(net, train_images, train_labels, 60000);
    
    // Evaluate on test data
    evaluate(net, test_images, test_labels, 10000);
    
    // Free memory
    freeNetwork(net);
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    
    return 0;
}


