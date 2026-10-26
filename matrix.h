#ifndef MATRIX_H
#define MATRIX_H

#define MAX_SIZE 27

typedef struct {
    int rows;
    int cols;
    double data[MAX_SIZE][MAX_SIZE];
} Matrix;

Matrix create_matrix(int rows, int cols);
Matrix multiply_matrices(Matrix A, Matrix B);
void print_matrix(Matrix A);

#endif