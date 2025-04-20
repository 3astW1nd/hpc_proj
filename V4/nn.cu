
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* ---------------- constants (unchanged) ----------------------------------- */
#define INPUT_SIZE   784
#define HIDDEN_SIZE  128
#define OUTPUT_SIZE  10
#define LEARNING_RATE 0.01f
#define EPOCHS       3
#define BATCH_SIZE   64
#define NUM_CLASSES  10

/* ---------------- CUDA helpers -------------------------------------------- */
#define CKCUDA(x)  { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(EXIT_FAILURE);} }
#define CKCUBLAS(x){ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS %s:%d error %d\n",__FILE__,__LINE__,int(s)); exit(EXIT_FAILURE);} }

/* ------------------ timing ------------------------------------------------ */
float elapsed(clock_t s){ return (float)(clock()-s)/CLOCKS_PER_SEC; }

/* ------------------ host matrix helpers (unchanged) ----------------------- */
float** allocMat(int r,int c){
    float** m=(float**)malloc(r*sizeof(float*));
    for(int i=0;i<r;++i) m[i]=(float*)malloc(c*sizeof(float));
    return m;
}
void freeMat(float** m,int r){ for(int i=0;i<r;++i) free(m[i]); free(m); }

/* ------------------ your original element‑wise kernels -------------------- */
__global__ void relu_gpu(float* x,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]=(x[i]>0.f)?x[i]:0.f;
}
__global__ void softmax_gpu(float* o){
    float m=o[0]; for(int i=1;i<OUTPUT_SIZE;++i) m=fmaxf(m,o[i]);
    float s=0.f;  for(int i=0;i<OUTPUT_SIZE;++i){ o[i]=expf(o[i]-m); s+=o[i]; }
    for(int i=0;i<OUTPUT_SIZE;++i) o[i]/=s;
}
/* output/hidden gradients + SGD kernels – unchanged */
__global__ void outputLayerGradient_gpu(float* d_out,const float* out,const float* y){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<OUTPUT_SIZE) d_out[i]=out[i]-y[i];
}
__global__ void hiddenLayerGradient_gpu(float* d_h,const float* W2,const float* d_out,const float* h){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<HIDDEN_SIZE){
        float s=0.f; for(int j=0;j<OUTPUT_SIZE;++j) s+=__ldg(&W2[j*HIDDEN_SIZE+i])*d_out[j];
        d_h[i]=s*(h[i]>0.f);
    }
}
__global__ void gradientDescentW_gpu(float* W,const float* d,const float* a,
                                     int rows,int cols,float lr){
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    int c=blockIdx.y*blockDim.y+threadIdx.y;
    if(r<rows && c<cols) W[r*cols+c]-=lr*d[r]*a[c];
}
__global__ void gradientDescentB_gpu(float* B,const float* d,int n,float lr){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) B[i]-=lr*d[i];
}

/* ------------------ structs ----------------------------------------------- */
typedef struct{
    float** W1; float** W2; float* b1; float* b2;
}NetCPU;

typedef struct{
    float* W1; float* W2; float* b1; float* b2;
    cublasHandle_t handle;                  /* NEW – one handle per GPU net */
}NetGPU;

/* ------------------ flatten / unflatten ----------------------------------- */
float* flatten(float** m,int r,int c){
    float* f=(float*)malloc(r*c*sizeof(float));
    for(int i=0;i<r;++i) memcpy(f+i*c,m[i],c*sizeof(float));
    return f;
}
void unflatten(const float* f,float** m,int r,int c){
    for(int i=0;i<r;++i) memcpy(m[i],f+i*c,c*sizeof(float));
}

/* ------------------ network creation -------------------------------------- */
NetCPU* createCPU(){
    NetCPU* n=(NetCPU*)malloc(sizeof(NetCPU));
    n->W1=allocMat(HIDDEN_SIZE,INPUT_SIZE);
    n->W2=allocMat(OUTPUT_SIZE,HIDDEN_SIZE);
    n->b1=(float*)calloc(HIDDEN_SIZE,sizeof(float));
    n->b2=(float*)calloc(OUTPUT_SIZE,sizeof(float));
    srand(0);
    for(int i=0;i<HIDDEN_SIZE;++i) for(int j=0;j<INPUT_SIZE;++j)
        n->W1[i][j]=((float)rand()/RAND_MAX)*0.01f;
    for(int i=0;i<OUTPUT_SIZE;++i) for(int j=0;j<HIDDEN_SIZE;++j)
        n->W2[i][j]=((float)rand()/RAND_MAX)*0.01f;
    return n;
}
NetGPU* createGPU(NetCPU* c){
    NetGPU* g=(NetGPU*)malloc(sizeof(NetGPU));
    float* fW1=flatten(c->W1,HIDDEN_SIZE,INPUT_SIZE);
    float* fW2=flatten(c->W2,OUTPUT_SIZE,HIDDEN_SIZE);
    CKCUDA(cudaMalloc(&g->W1,HIDDEN_SIZE*INPUT_SIZE*sizeof(float)));
    CKCUDA(cudaMalloc(&g->W2,OUTPUT_SIZE*HIDDEN_SIZE*sizeof(float)));
    CKCUDA(cudaMalloc(&g->b1,HIDDEN_SIZE*sizeof(float)));
    CKCUDA(cudaMalloc(&g->b2,OUTPUT_SIZE*sizeof(float)));
    CKCUDA(cudaMemcpy(g->W1,fW1,HIDDEN_SIZE*INPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice));
    CKCUDA(cudaMemcpy(g->W2,fW2,OUTPUT_SIZE*HIDDEN_SIZE*sizeof(float),cudaMemcpyHostToDevice));
    CKCUDA(cudaMemcpy(g->b1,c->b1,HIDDEN_SIZE*sizeof(float),cudaMemcpyHostToDevice));
    CKCUDA(cudaMemcpy(g->b2,c->b2,OUTPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice));
    free(fW1); free(fW2);
    CKCUBLAS(cublasCreate(&g->handle));
    CKCUBLAS(cublasSetMathMode(g->handle,CUBLAS_TENSOR_OP_MATH));   /* TF32 / tensor cores */
    return g;
}
void gpuToCPU(NetGPU* g,NetCPU* c){
    float* fW1=(float*)malloc(HIDDEN_SIZE*INPUT_SIZE*sizeof(float));
    float* fW2=(float*)malloc(OUTPUT_SIZE*HIDDEN_SIZE*sizeof(float));
    CKCUDA(cudaMemcpy(fW1,g->W1,HIDDEN_SIZE*INPUT_SIZE*sizeof(float),cudaMemcpyDeviceToHost));
    CKCUDA(cudaMemcpy(fW2,g->W2,OUTPUT_SIZE*HIDDEN_SIZE*sizeof(float),cudaMemcpyDeviceToHost));
    CKCUDA(cudaMemcpy(c->b1,g->b1,HIDDEN_SIZE*sizeof(float),cudaMemcpyDeviceToHost));
    CKCUDA(cudaMemcpy(c->b2,g->b2,OUTPUT_SIZE*sizeof(float),cudaMemcpyDeviceToHost));
    unflatten(fW1,c->W1,HIDDEN_SIZE,INPUT_SIZE);
    unflatten(fW2,c->W2,OUTPUT_SIZE,HIDDEN_SIZE);
    free(fW1); free(fW2);
}
void freeGPU(NetGPU* g){
    cublasDestroy(g->handle);
    cudaFree(g->W1); cudaFree(g->W2); cudaFree(g->b1); cudaFree(g->b2); free(g);
}

/* ------------------ forward / backward (tensor‑core SGEMM) ---------------- */
__host__ inline void gemm_vector(const float* A,int rows,int cols,
                                 const float* x,float* y,cublasHandle_t h){
    /* y = A * x   (A row‑major, dims rows×cols ,  x length cols, y length rows)
       trick: treat A as (cols×rows) column‑major and use opT */
    const float alpha=1.f,beta=0.f;
    CKCUBLAS(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                         rows, 1, cols,
                         &alpha, A, cols,      /* lda = cols because A is row‑major */
                         x, cols,
                         &beta,  y, rows));
}

void forwardGPU(NetGPU* n,float* d_in,float* d_h,float* d_out,cudaStream_t st){
    cublasSetStream(n->handle,st);
    gemm_vector(n->W1,HIDDEN_SIZE,INPUT_SIZE,d_in,d_h,n->handle);
    int th=256,bl=(HIDDEN_SIZE+th-1)/th;
    relu_gpu<<<bl,th,0,st>>>(d_h,HIDDEN_SIZE);
    gemm_vector(n->W2,OUTPUT_SIZE,HIDDEN_SIZE,d_h,d_out,n->handle);
    softmax_gpu<<<1,1,0,st>>>(d_out);
    /* add biases on host side of original code? keep kernels if needed */
}

void backwardGPU(NetGPU* n,float* d_in,float* d_h,float* d_out,float* d_tgt,cudaStream_t st){
    cublasSetStream(n->handle,st);
    float *d_do,*d_dh; CKCUDA(cudaMalloc(&d_do,OUTPUT_SIZE*sizeof(float)));
    CKCUDA(cudaMalloc(&d_dh,HIDDEN_SIZE*sizeof(float)));

    int th=256,og=(OUTPUT_SIZE+th-1)/th,hg=(HIDDEN_SIZE+th-1)/th;
    outputLayerGradient_gpu<<<og,th,0,st>>>(d_do,d_out,d_tgt);
    hiddenLayerGradient_gpu<<<hg,th,0,st>>>(d_dh,n->W2,d_do,d_h);

    dim3 OB(16,16); dim3 OG((OUTPUT_SIZE+OB.x-1)/OB.x,(HIDDEN_SIZE+OB.y-1)/OB.y);
    dim3 HB(16,16); dim3 HG((HIDDEN_SIZE+HB.x-1)/HB.x,(INPUT_SIZE+HB.y-1)/HB.y);

    gradientDescentW_gpu<<<OG,OB,0,st>>>(n->W2,d_do,d_h,OUTPUT_SIZE,HIDDEN_SIZE,LEARNING_RATE);
    gradientDescentB_gpu<<<og,th,0,st>>>(n->b2,d_do,OUTPUT_SIZE,LEARNING_RATE);
    gradientDescentW_gpu<<<HG,HB,0,st>>>(n->W1,d_dh,d_in,HIDDEN_SIZE,INPUT_SIZE,LEARNING_RATE);
    gradientDescentB_gpu<<<hg,th,0,st>>>(n->b1,d_dh,HIDDEN_SIZE,LEARNING_RATE);

    cudaFree(d_do); cudaFree(d_dh);
}

/* ------------------ MNIST I/O (with fread check) -------------------------- */
float** loadImg(const char* p,int n){
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p); exit(1);}
    fseek(f,16,SEEK_SET);
    float** a=allocMat(n,INPUT_SIZE);
    for(int i=0;i<n;++i) for(int j=0;j<INPUT_SIZE;++j){
        unsigned char px; if(fread(&px,1,1,f)!=1){ fprintf(stderr,"EOF %s\n",p); exit(EXIT_FAILURE);}
        a[i][j]=px/255.f;
    }
    fclose(f); return a;
}
float** loadLbl(const char* p,int n){
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p); exit(1);}
    fseek(f,8,SEEK_SET);
    float** y=allocMat(n,OUTPUT_SIZE);
    for(int i=0;i<n;++i){
        unsigned char l; if(fread(&l,1,1,f)!=1){fprintf(stderr,"EOF %s\n",p); exit(EXIT_FAILURE);}
        for(int j=0;j<OUTPUT_SIZE;++j) y[i][j]=(j==l)?1.f:0.f;
    }
    fclose(f); return y;
}

/* ------------------ training & evaluation (your original loops) ----------- */
void train(NetCPU* cpu,float** x,float** y,int n){
    NetGPU* g=createGPU(cpu);
    float *d_in,*d_h,*d_out,*d_y;
    CKCUDA(cudaMalloc(&d_in,INPUT_SIZE*sizeof(float)));
    CKCUDA(cudaMalloc(&d_h,HIDDEN_SIZE*sizeof(float)));
    CKCUDA(cudaMalloc(&d_out,OUTPUT_SIZE*sizeof(float)));
    CKCUDA(cudaMalloc(&d_y,OUTPUT_SIZE*sizeof(float)));
    float* h_out; CKCUDA(cudaHostAlloc(&h_out,OUTPUT_SIZE*sizeof(float),0));
    cudaStream_t st; CKCUDA(cudaStreamCreate(&st));

    clock_t t0=clock();
    for(int e=0;e<EPOCHS;++e){
        float loss=0.f; int correct=0; clock_t es=clock();
        for(int i=0;i<n;++i){
            CKCUDA(cudaMemcpyAsync(d_in,x[i],INPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice,st));
            CKCUDA(cudaMemcpyAsync(d_y ,y[i],OUTPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice,st));
            forwardGPU(g,d_in,d_h,d_out,st);
            backwardGPU(g,d_in,d_h,d_out,d_y,st);
            CKCUDA(cudaMemcpyAsync(h_out,d_out,OUTPUT_SIZE*sizeof(float),cudaMemcpyDeviceToHost,st));
            CKCUDA(cudaStreamSynchronize(st));
            for(int k=0;k<OUTPUT_SIZE;++k) loss-=y[i][k]*logf(fmaxf(h_out[k],1e-8f));
            int p=0,a=0;
            for(int j=1;j<OUTPUT_SIZE;++j){ if(h_out[j]>h_out[p]) p=j; if(y[i][j]>y[i][a]) a=j; }
            if(p==a) ++correct;
        }
        printf("Epoch %d – Loss: %.4f – Train Accuracy %.2f%% – Time: %.2fs\n",
               e+1,loss/n,100.f*correct/(float)n,elapsed(es));
    }
    printf("Total training time: %.2fs\n",elapsed(t0));
    gpuToCPU(g,cpu);
    cudaFreeHost(h_out); cudaFree(d_in); cudaFree(d_h); cudaFree(d_out); cudaFree(d_y);
    cudaStreamDestroy(st); freeGPU(g);
}

void evaluate(NetCPU* cpu,float** x,float** y,int n){
    NetGPU* g=createGPU(cpu);
    float *d_in,*d_h,*d_out; CKCUDA(cudaMalloc(&d_in,INPUT_SIZE*sizeof(float)));
    CKCUDA(cudaMalloc(&d_h,HIDDEN_SIZE*sizeof(float))); CKCUDA(cudaMalloc(&d_out,OUTPUT_SIZE*sizeof(float)));
    float* h_out; CKCUDA(cudaHostAlloc(&h_out,OUTPUT_SIZE*sizeof(float),0));
    cudaStream_t st; CKCUDA(cudaStreamCreate(&st));
    int correct=0;
    for(int i=0;i<n;++i){
        CKCUDA(cudaMemcpyAsync(d_in,x[i],INPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice,st));
        forwardGPU(g,d_in,d_h,d_out,st);
        CKCUDA(cudaMemcpyAsync(h_out,d_out,OUTPUT_SIZE*sizeof(float),cudaMemcpyDeviceToHost,st));
        CKCUDA(cudaStreamSynchronize(st));
        int p=0,a=0; for(int j=1;j<OUTPUT_SIZE;++j){
            if(h_out[j]>h_out[p]) p=j; if(y[i][j]>y[i][a]) a=j;}
        if(p==a) ++correct;
    }
    printf("Test Accuracy: %.2f%%\n",100.f*correct/(float)n);
    cudaFreeHost(h_out); cudaFree(d_in); cudaFree(d_h); cudaFree(d_out);
    cudaStreamDestroy(st); freeGPU(g);
}

/* ------------------ main -------------------------------------------------- */
int main(){
    printf("MNIST Neural Network - Tensor Core Implementatino (V4)\n");
    float** train_x=loadImg("../data/train-images.idx3-ubyte",60000);
    float** train_y=loadLbl("../data/train-labels.idx1-ubyte",60000);
    float** test_x =loadImg("../data/t10k-images.idx3-ubyte",10000);
    float** test_y =loadLbl("../data/t10k-labels.idx1-ubyte",10000);

    NetCPU* net=createCPU();
    train(net,train_x,train_y,60000);
    evaluate(net,test_x,test_y,10000);

    freeMat(train_x,60000); freeMat(train_y,60000);
    freeMat(test_x,10000);  freeMat(test_y,10000);
    freeMat(net->W1,HIDDEN_SIZE); freeMat(net->W2,OUTPUT_SIZE);
    free(net->b1); free(net->b2); free(net);
    return 0;
}
