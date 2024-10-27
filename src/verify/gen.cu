#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>
#include <math.h>
#include <fstream>
/**
 * Kernel config
 */
#define BLOCK_SIZE 256
#define COOPERATE_BLOCK_NUM 32 // number of blocks working on a request (>HEAD_NUM)

/**
 * Decode config
 */
#define BATCH_SIZE 1
#define HEAD_DIM 64        // attn head dimension
#define HEAD_NUM 2      // attn head number
#define SMALL_COOPERATE_BLOCK_NUM (COOPERATE_BLOCK_NUM/HEAD_NUM) // 16

#define FFN_HIDDEN 64     // ffn hidden dimension
#define EMBEDDING_DIM 64  // token embedding dimension
#define SEQ_LEN 8        // sequence length
#define FL_DEC_SPLIT 8
#define WORKLOAD FL_DEC_SPLIT*((SEQ_LEN+(FL_DEC_SPLIT*SMALL_COOPERATE_BLOCK_NUM)-1)/(FL_DEC_SPLIT*SMALL_COOPERATE_BLOCK_NUM))

std::mt19937 rng(42);
std::normal_distribution<float> norm_dist(0.0, 1.0);

template<typename T>
void fill_matrix(T *mat, int sz) {
    for (int i = 0; i < sz; i++) {
        float random_value;
        do {
            random_value = norm_dist(rng);
        } while (random_value < -1 || random_value > 1);
        if constexpr(std::is_same<T, __half>::value)
        {
            mat[i] = __float2half(random_value); // convert needed
        } else {
            mat[i] = random_value;
        }
    }
}

template<typename T>
void writeDataToFile(const T *data, int size, const std::string &filename) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    for (int i = 0; i < size; ++i) {
        if constexpr(std::is_same<T, __half>::value)
        {
            outfile << __half2float(data[i]) << "\t";
        } else {
            outfile << data[i] << "\t";
        }
    }
    outfile.close();
}

int main(int argc, char **argv) {

    half *h_input;
    half *h_k_cache;
    half *h_v_cache;
    half *h_w_q;
    half *h_w_k;
    half *h_w_v;
    half *h_w_o;
    half *h_ffn_1;
    half *h_ffn_2;
    half *h_ffn_3;
    half *h_rms_1;
    half *h_rms_2;

    h_input = new half[BATCH_SIZE * 1 * EMBEDDING_DIM];
    h_w_q = new half[HEAD_NUM * EMBEDDING_DIM * HEAD_DIM];
    h_w_k = new half[HEAD_NUM * EMBEDDING_DIM * HEAD_DIM];
    h_w_v = new half[HEAD_NUM * EMBEDDING_DIM * HEAD_DIM];
    h_w_o = new half[HEAD_NUM * HEAD_DIM * EMBEDDING_DIM];
    h_k_cache = new half[BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM];
    h_v_cache = new half[BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM];
    h_ffn_1 = new half[EMBEDDING_DIM * FFN_HIDDEN];
    h_ffn_2 = new half[FFN_HIDDEN * EMBEDDING_DIM];
    h_ffn_3 = new half[EMBEDDING_DIM * FFN_HIDDEN];
    h_rms_1 = new half[EMBEDDING_DIM];
    h_rms_2 = new half[EMBEDDING_DIM];

    fill_matrix(h_input, BATCH_SIZE * 1 * EMBEDDING_DIM);
    fill_matrix(h_w_q, HEAD_NUM * EMBEDDING_DIM * HEAD_DIM);
    fill_matrix(h_w_k, HEAD_NUM * EMBEDDING_DIM * HEAD_DIM);
    fill_matrix(h_w_v, HEAD_NUM * EMBEDDING_DIM * HEAD_DIM);
    fill_matrix(h_w_o, HEAD_NUM * HEAD_DIM * EMBEDDING_DIM);
    fill_matrix(h_k_cache, BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM);
    fill_matrix(h_v_cache, BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM);
    fill_matrix(h_ffn_1, EMBEDDING_DIM * FFN_HIDDEN);
    fill_matrix(h_ffn_2, FFN_HIDDEN * EMBEDDING_DIM);
    fill_matrix(h_ffn_3, EMBEDDING_DIM * FFN_HIDDEN);
    fill_matrix(h_rms_1, EMBEDDING_DIM);
    fill_matrix(h_rms_2, EMBEDDING_DIM);

    writeDataToFile(h_input, BATCH_SIZE * 1 * EMBEDDING_DIM, "data/h_input");

    writeDataToFile(h_w_q, HEAD_NUM * EMBEDDING_DIM * HEAD_DIM, "data/h_w_q");
    writeDataToFile(h_w_k, HEAD_NUM * EMBEDDING_DIM * HEAD_DIM, "data/h_w_k");
    writeDataToFile(h_w_v, HEAD_NUM * EMBEDDING_DIM * HEAD_DIM, "data/h_w_v");

    // writeDataToFile(h_w_o, HEAD_NUM * HEAD_DIM * EMBEDDING_DIM, "data/h_w_o");
    std::ofstream outfile("data/h_w_o");
    if (!outfile) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    for (int b = 0; b < EMBEDDING_DIM; ++b) {
        for (int j = 0; j < HEAD_NUM; ++j) {
            for (int k = 0; k < HEAD_DIM; ++k) {
                outfile << __half2float(h_w_o[
                                                j * HEAD_DIM * EMBEDDING_DIM + k + b * HEAD_DIM
                                        ]) << "\t";
            }
            outfile << "\n";
        }
        outfile << "\n";
    }
    outfile.close();

    // writeDataToFile(h_k_cache, BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM, "data/h_k_cache");
    std::ofstream outfile_h_k_cache("data/h_k_cache");
    if (!outfile_h_k_cache) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int j = 0; j < HEAD_NUM; ++j) {
            for (int i = 0; i < SEQ_LEN; ++i) {
                for (int k = 0; k < HEAD_DIM; ++k) {
                    outfile_h_k_cache << __half2float(h_k_cache[
                                                              b * HEAD_NUM * SEQ_LEN * HEAD_DIM
                                                              + j * SEQ_LEN * HEAD_DIM + i * HEAD_DIM + k
                                                      ]) << "\t";
                }
            }
            outfile_h_k_cache << "\n";
        }
        outfile_h_k_cache << "\n";
    }
    outfile_h_k_cache.close();
    // writeDataToFile(h_v_cache, BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM, "data/h_v_cache");
    std::ofstream outfile_h_v_cache("data/h_v_cache");
    if (!outfile_h_v_cache) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int j = 0; j < HEAD_NUM; ++j) {
            for (int i = 0; i < SEQ_LEN; ++i) {
                for (int k = 0; k < HEAD_DIM; ++k) {
                    outfile_h_v_cache << __half2float(h_v_cache[
                                                              b * HEAD_NUM * SEQ_LEN * HEAD_DIM +
                                                              j * SEQ_LEN * HEAD_DIM + i + k * SEQ_LEN
                                                      ]) << "\t";
                }
            }
            outfile_h_v_cache << "\n";
        }
        outfile_h_v_cache << "\n";
    }
    outfile_h_v_cache.close();

    std::ofstream outfile_1("data/h_ffn_1");
    if (!outfile_1) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    for (int b = 0; b < EMBEDDING_DIM; ++b) {
        for (int j = 0; j < FFN_HIDDEN; ++j) {
            outfile_1 << __half2float(h_ffn_1[j * EMBEDDING_DIM + b]) << "\t";
        }
        outfile_1 << "\n";
    }
    outfile_1.close();

    writeDataToFile(h_ffn_2, FFN_HIDDEN * EMBEDDING_DIM, "data/h_ffn_2");
    std::ofstream outfile_3("data/h_ffn_3");
    if (!outfile_3) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    for (int b = 0; b < EMBEDDING_DIM; ++b) {
        for (int j = 0; j < FFN_HIDDEN; ++j) {
            outfile_3 << __half2float(h_ffn_3[j * EMBEDDING_DIM + b]) << "\t";
        }
        outfile_3 << "\n";
    }
    outfile_3.close();

    writeDataToFile(h_rms_1, EMBEDDING_DIM, "data/h_rms_1");
    writeDataToFile(h_rms_2, EMBEDDING_DIM, "data/h_rms_2");
}