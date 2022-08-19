# CUDA-k-mer-counting

By - Anuradha Wickramarachchi

## Compile code

```bash
nvcc k-mer-counting-improved.cu -o kmercountingimproved -lz --dopt=on -gencode arch=compute_80,code=sm_80 -Xcompiler -fopenmp -O3
```

## Docs and refs

* intro https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
* possible optimizations https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf

