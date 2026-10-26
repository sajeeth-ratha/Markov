
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

Matrix create_matrix(int rows, int cols)
{
    if (rows < 1 || rows > MAX_SIZE ||
        cols < 1 || cols > MAX_SIZE) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }

    Matrix M = {0};
    M.rows = rows;
    M.cols = cols;
    return M;
}

Matrix multiply_matrices(Matrix A, Matrix B)
{
    if (A.cols != B.rows) {
        fprintf(stderr, "Incompatible matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }

    Matrix C = create_matrix(A.rows, B.cols);

    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            for (int k = 0; k < A.cols; k++) {
                C.data[i][j] += A.data[i][k] * B.data[k][j];
            }
        }
    }

    return C;
}

void print_matrix(Matrix A)
{
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            printf("%.6f ", A.data[i][j]);
        }
        printf("\n");
    }
}
