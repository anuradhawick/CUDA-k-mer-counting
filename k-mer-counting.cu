/*
Copyright 2022 Anuradha Wickramarachchi (anuradhawick@gmail.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <zlib.h>
#include <iostream>
#include <omp.h>
#include <atomic>
#include <vector>
#include <algorithm>

#include "kseq.h"

using namespace std;

KSEQ_INIT(gzFile, gzread)
const uint64_t K_SIZE = 15;
const uint64_t K_MER_COUNT = (uint64_t)pow(4, K_SIZE);
const uint64_t K_MER_MASK = (uint64_t)pow(4, K_SIZE) - 1;
// const char *GENOME = "Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz";
const char *GENOME = "GRCH38.fasta";
// const char *GENOME = "/media/anuvini/98C4876BC4874B08/lrb_extension/set_100/reads.fasta";


__global__ void CUDA_count_k_mers(uint32_t *k_mer_counts, char *seq, uint64_t len)
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len - K_SIZE)
    {
        uint64_t val = 0;
        for (size_t j = 0; j < K_SIZE; j++)
        {
            char s = seq[i + j];
            if (s >= 97 && s <= 122)
            {
                s = s - 32;
            }
            const uint64_t bits = ((s >> 1) & 3);
            if (s == 'A' || s == 'C' || s == 'G' || s == 'T')
            {
                val <<= 2;
                val += bits;
            }
            else
            {
                break;
            }

            if (j == K_SIZE - 1)
            {
                atomicAdd(&k_mer_counts[val], 1);
            }
        }
    }
}

void count_k_mers(vector<atomic<uint32_t>> &k_mer_counts, const char *seq, const size_t &len)
{
#pragma omp parallel for num_threads(32)
    for (size_t i = 0; i < len - K_SIZE; i++)
    {
        uint64_t val = 0;
        for (size_t j = 0; j < K_SIZE; j++)
        {
            const char s = toupper(seq[i + j]);
            const uint64_t bits = ((s >> 1) & 3);
            if (s == 'A' || s == 'C' || s == 'G' || s == 'T')
            {
                val <<= 2;
                val += bits;
            }
            else
            {
                break;
            }

            if (j == K_SIZE - 1)
            {
                uint32_t oval = k_mer_counts[val];
                while (!k_mer_counts[val].compare_exchange_weak(oval, oval + (uint32_t)1))
                {
                };
            }
        }
    }
}

// nvcc k-mer-counting.cu -o kmercounting -lz --dopt=on -gencode arch=compute_80,code=sm_80 -Xcompiler -fopenmp -O3
// intro https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
// possible optimizations https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf
int main(int argc, char **argv)
{
    gzFile fp;
    kseq_t *seq;
    uint64_t count = 0;
    vector<uint32_t> cpu_k_mer_counts;
    uint32_t *gpu_k_mer_counts;

    if (argc == 2 && string(argv[1]) == "assert")
    {
        cout << "Running the test case" << endl;
    }

    if (argc == 1 || string(argv[1]) == "cpu" || string(argv[1]) == "assert")
    {
        cout << "Running CPU mode" << endl;
        vector<atomic<uint32_t>> k_mer_counts(K_MER_COUNT);

        fp = gzopen(GENOME, "r");
        seq = kseq_init(fp);

        while (kseq_read(seq) >= 0)
        {
            count_k_mers(k_mer_counts, seq->seq.s, seq->seq.l);
            cout << "Counted : " << ++count << "                 \r" << flush;
        }
        cout << "Counted : " << count << endl;

        if ((argc == 2 && string(argv[1]) == "assert"))
        {
            transform(k_mer_counts.begin(), k_mer_counts.end(), back_inserter(cpu_k_mer_counts), [](atomic<uint32_t> &i)
                      { return i.load(memory_order_relaxed); });
        }

        kseq_destroy(seq);
        gzclose(fp);
    }
    if (argc == 2 && string(argv[1]) == "cuda" || argc == 2 && string(argv[1]) == "assert")
    {
        count = 0;
        cout << "Running CUDA mode" << endl;
        uint32_t *k_mer_counts, *cuda_k_mer_counts;
        k_mer_counts = (uint32_t *)malloc(K_MER_COUNT * sizeof(uint32_t));
        for (size_t i = 0; i < K_MER_COUNT; i++)
        {
            k_mer_counts[i] = 0;
        }
        cudaMalloc((void **)&cuda_k_mer_counts, K_MER_COUNT * sizeof(uint32_t));
        cudaMemcpy(cuda_k_mer_counts, k_mer_counts, K_MER_COUNT * sizeof(uint32_t), cudaMemcpyHostToDevice);

        fp = gzopen(GENOME, "r");
        seq = kseq_init(fp);
        char *cuda_seq;
        cudaMalloc((void **)&cuda_seq, 1024 * 1024 * 1024 * sizeof(char));

        while (kseq_read(seq) >= 0)
        {
            cudaMemcpy(cuda_seq, seq->seq.s, seq->seq.l * sizeof(char), cudaMemcpyHostToDevice);
            CUDA_count_k_mers<<<1024 * 1024, 1024>>>(cuda_k_mer_counts, cuda_seq, seq->seq.l);
            cout << "Counted : " << ++count << "                 \r" << flush;
        }
        cudaFree(cuda_seq);
        cudaMemcpy(k_mer_counts, cuda_k_mer_counts, K_MER_COUNT * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cout << "Counted : " << count << endl;

        kseq_destroy(seq);
        gzclose(fp);
        if ((argc == 2 && string(argv[1]) == "assert"))
        {
            gpu_k_mer_counts = k_mer_counts;
        }
        else
        {
            free(k_mer_counts);
        }
    }
    if (argc == 2 && string(argv[1]) == "assert")
    {
        for (size_t i = 0; i < K_MER_COUNT; i++)
        {
            if (gpu_k_mer_counts[i] != cpu_k_mer_counts[i])
            {
                cout << i << " " << gpu_k_mer_counts[i] << " " << cpu_k_mer_counts[i] << endl;
                cout << "Operation Failed" << endl;
                return -1;
            }
        }
        cout << "Test passed" << endl;
        free(gpu_k_mer_counts);
    }
    return 0;
}