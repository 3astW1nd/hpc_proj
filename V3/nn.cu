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


#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// GPU network structure
typedef struct {
    double* W1;  // Flattened matrix
    double* W2;  // Flattened matrix
    double* b1;
    double* b2;
} NeuralNetworkGPU;

// ReLU activation for GPU
__global__ void relu_gpu(double* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

// Softmax activation for GPU
__global__ void softmax_gpu(double* output, int outputSize) {
    // First find the maximum value for numerical stability
    double maxVal = output[0];
    for (int i = 1; i < outputSize; i++) {
        if (output[i] > maxVal) {
            maxVal = output[i];
        }
    }
    
    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (int i = 0; i < outputSize; i++) {
        output[i] = exp(output[i] - maxVal);
        sum += output[i];
    }
    
    // Normalize
    for (int i = 0; i < outputSize; i++) {
        output[i] /= sum;
    }
}

// Matrix multiplication for GPU
__global__ void matrixMulGpu(double* W, double* input, double* res, double* b, int size1, int size2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size1) {
        double sum = b[i];
        for (int j = 0; j < size2; j++) {
            sum += W[i * size2 + j] * input[j];
        }
        res[i] = sum;
    }
}

// Output layer gradient computation for GPU
__global__ void outputLayerGradient_gpu(double* d_output, double* output, double* target) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < OUTPUT_SIZE) {
        d_output[i] = output[i] - target[i];
    }
}

// Hidden layer gradient computation for GPU
__global__ void hiddenLayerGradient_gpu(double* d_hidden, double* W2, double* d_output, double* hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < HIDDEN_SIZE) {
        double sum = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sum += W2[j * HIDDEN_SIZE + i] * d_output[j];
        }
        d_hidden[i] = sum * (hidden[i] > 0); // ReLU derivative
    }
}

// Weight gradient descent for GPU
__global__ void gradientDescentW_gpu(double* W, double* d1, double* arr, int size1, int size2, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < size1 && j < size2) {
        W[i * size2 + j] -= learning_rate * d1[i] * arr[j];
    }
}

// Bias gradient descent for GPU
__global__ void gradientDescentB_gpu(double* B, double* d1, int size, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        B[i] -= learning_rate * d1[i];
    }
}

// 2D to 1D array flattening
double* flattenMatrix(double** mat, int rows, int cols) {
    double* flat = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = mat[i][j];
        }
    }
    return flat;
}

// 1D to 2D array unflattening
void unflattenMatrix(double* flat, double** mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = flat[i * cols + j];
        }
    }
}

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

// Initialize GPU network from CPU network
NeuralNetworkGPU* createNetworkGPU(NeuralNetwork* cpuNet) {
    NeuralNetworkGPU* gpuNet = (NeuralNetworkGPU*)malloc(sizeof(NeuralNetworkGPU));
    
    // Flatten the weight matrices
    double* flat_W1 = flattenMatrix(cpuNet->W1, HIDDEN_SIZE, INPUT_SIZE);
    double* flat_W2 = flattenMatrix(cpuNet->W2, OUTPUT_SIZE, HIDDEN_SIZE);
    
    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->b2, OUTPUT_SIZE * sizeof(double)));
    
    // Copy data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->W1, flat_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->W2, flat_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->b1, cpuNet->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->b2, cpuNet->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
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
    // Allocate temporary arrays for flattened weights
    double* flat_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* flat_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    // Copy weights from GPU
    CHECK_CUDA_ERROR(cudaMemcpy(flat_W1, gpuNet->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(flat_W2, gpuNet->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Copy biases from GPU
    CHECK_CUDA_ERROR(cudaMemcpy(cpuNet->b1, gpuNet->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(cpuNet->b2, gpuNet->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Unflatten weights
    unflattenMatrix(flat_W1, cpuNet->W1, HIDDEN_SIZE, INPUT_SIZE);
    unflattenMatrix(flat_W2, cpuNet->W2, OUTPUT_SIZE, HIDDEN_SIZE);
    
    free(flat_W1);
    free(flat_W2);
}

// Forward pass on GPU
void forwardGPU(NeuralNetworkGPU* net, double* d_input, double* d_hidden, double* d_output) {
    // Define block and grid dimensions
    int blockSize = 256;
    int hiddenGridSize = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    int outputGridSize = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    
    // Compute hidden layer activations
    matrixMulGpu<<<hiddenGridSize, blockSize>>>(net->W1, d_input, d_hidden, net->b1, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Apply ReLU activation to hidden layer
    relu_gpu<<<hiddenGridSize, blockSize>>>(d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Compute output layer pre-activations
    matrixMulGpu<<<outputGridSize, blockSize>>>(net->W2, d_hidden, d_output, net->b2, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Apply softmax activation
    softmax_gpu<<<1, 1>>>(d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Backward pass on GPU
void backwardGPU(NeuralNetworkGPU* net, double* d_input, double* d_hidden, double* d_output, double* d_target) {
    // Allocate memory for gradients
    double *d_d_output, *d_d_hidden;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_hidden, HIDDEN_SIZE * sizeof(double)));
    
    // Define block and grid dimensions
    int blockSize = 256;
    int outputGridSize = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    int hiddenGridSize = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    
    // Compute output layer gradients
    outputLayerGradient_gpu<<<outputGridSize, blockSize>>>(d_d_output, d_output, d_target);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Compute hidden layer gradients
    hiddenLayerGradient_gpu<<<hiddenGridSize, blockSize>>>(d_d_hidden, net->W2, d_d_output, d_hidden);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Define block and grid dimensions for weight updates
    dim3 outputBlockDim(16, 16);
    dim3 outputGridDim((OUTPUT_SIZE + outputBlockDim.x - 1) / outputBlockDim.x, 
                      (HIDDEN_SIZE + outputBlockDim.y - 1) / outputBlockDim.y);
    
    dim3 hiddenBlockDim(16, 16);
    dim3 hiddenGridDim((HIDDEN_SIZE + hiddenBlockDim.x - 1) / hiddenBlockDim.x, 
                      (INPUT_SIZE + hiddenBlockDim.y - 1) / hiddenBlockDim.y);
    
    // Update output layer weights and biases
    gradientDescentW_gpu<<<outputGridDim, outputBlockDim>>>(net->W2, d_d_output, d_hidden, 
                                                          OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    gradientDescentB_gpu<<<outputGridSize, blockSize>>>(net->b2, d_d_output, OUTPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Update hidden layer weights and biases
    gradientDescentW_gpu<<<hiddenGridDim, hiddenBlockDim>>>(net->W1, d_d_hidden, d_input, 
                                                         HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    gradientDescentB_gpu<<<hiddenGridSize, blockSize>>>(net->b1, d_d_hidden, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Free temporary memory
    CHECK_CUDA_ERROR(cudaFree(d_d_output));
    CHECK_CUDA_ERROR(cudaFree(d_d_hidden));
}

// Train network
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    
    // Create GPU network
    NeuralNetworkGPU* gpuNet = createNetworkGPU(net);
    
    // Allocate GPU memory for input, hidden, output, and target
    double *d_input, *d_hidden, *d_output, *d_target;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_target, OUTPUT_SIZE * sizeof(double)));
    
    // Allocate host memory for results
    double* h_output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        
        for (int i = 0; i < numImages; i++) {
            // Copy input and target to GPU
            CHECK_CUDA_ERROR(cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_target, labels[i], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
            
            // Forward pass
            forwardGPU(gpuNet, d_input, d_hidden, d_output);
            
            // Backward pass
            backwardGPU(gpuNet, d_input, d_hidden, d_output, d_target);
            
            // Copy output back to host for loss and accuracy calculation
            CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
            
            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= labels[i][k] * log(h_output[k] > 1e-10 ? h_output[k] : 1e-10);
            }
            
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output[j] > h_output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    
    printf("Total training time: %.3fs\n", get_time(total_start));
    
    // Copy final weights back to CPU
    copyNetworkToCPU(gpuNet, net);
    
    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_target));
    freeNetworkGPU(gpuNet);
    free(h_output);
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    // Create GPU network
    NeuralNetworkGPU* gpuNet = createNetworkGPU(net);
    
    // Allocate GPU memory
    double *d_input, *d_hidden, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(double)));
    
    // Allocate host memory for results
    double* h_output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        // Copy input to GPU
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
        
        // Forward pass
        forwardGPU(gpuNet, d_input, d_hidden, d_output);
        
        // Copy output back to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
    
    // Free memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    freeNetworkGPU(gpuNet);
    free(h_output);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
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

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
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
    printf("MNIST Neural Network - GPU Implementation (V2)\n\n");
    
    // Load MNIST dataset
    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);
    
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



