#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_TRANSIENT 4
#define N_CLASSES 3

// States ordering:
// Transient: [0: State 10, 1: State 19, 2: State 22, 3: State 24]
// Persistent Classes: [0: Class A, 1: Class B, 2: Class C3]

void invert_matrix(double A[N_TRANSIENT][N_TRANSIENT], double inv[N_TRANSIENT][N_TRANSIENT]) {
    double aug[N_TRANSIENT][2 * N_TRANSIENT];
    
    // Create augmented matrix [A | I]
    for (int i = 0; i < N_TRANSIENT; i++) {
        for (int j = 0; j < N_TRANSIENT; j++) {
            aug[i][j] = A[i][j];
            aug[i][j + N_TRANSIENT] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Perform Gauss-Jordan elimination
    for (int i = 0; i < N_TRANSIENT; i++) {
        double pivot = aug[i][i];
        for (int j = 0; j < 2 * N_TRANSIENT; j++) {
            aug[i][j] /= pivot;
        }
        for (int k = 0; k < N_TRANSIENT; k++) {
            if (k != i) {
                double factor = aug[k][i];
                for (int j = 0; j < 2 * N_TRANSIENT; j++) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }
    
    // Extract inverse
    for (int i = 0; i < N_TRANSIENT; i++) {
        for (int j = 0; j < N_TRANSIENT; j++) {
            inv[i][j] = aug[i][j + N_TRANSIENT];
        }
    }
}

int main() {
    // Transition probabilities between transient states Q (4x4)
    // Rows: 10, 19, 22, 24
    double Q[N_TRANSIENT][N_TRANSIENT] = {
        {0.0, 0.0, 0.0, 0.0}, // State 10
        {0.0, 0.0, 0.0, 0.0}, // State 19
        {0.0, 0.0, 0.0, 0.0}, // State 22
        {0.0, 0.0, 0.0, 0.1}  // State 24 (0.1 self-loop)
    };
    
    // Transition probabilities from transient states to persistent classes R (4x3)
    // Columns: Class A, Class B, Class C3
    double R[N_TRANSIENT][N_CLASSES] = {
        {0.5, 0.0, 0.5}, // State 10 -> (0.5 to A, 0.0 to B, 0.5 to C3)
        {0.5, 0.5, 0.0}, // State 19 -> (0.5 to A, 0.5 to B, 0.0 to C3)
        {1.0/3.0, 2.0/3.0, 0.0}, // State 22 -> (1/3 to A, 2/3 to B, 0.0 to C3)
        {0.4, 0.0, 0.5}  // State 24 -> (0.4 to A, 0.0 to B, 0.5 to C3)
    };
    
    // Compute (I - Q)
    double IQ[N_TRANSIENT][N_TRANSIENT];
    for (int i = 0; i < N_TRANSIENT; i++) {
        for (int j = 0; j < N_TRANSIENT; j++) {
            IQ[i][j] = (i == j ? 1.0 : 0.0) - Q[i][j];
        }
    }
    
    // Compute Fundamental Matrix M = (I - Q)^(-1)
    double M[N_TRANSIENT][N_TRANSIENT];
    invert_matrix(IQ, M);
    
    // Compute Absorption Probabilities B = M * R
    double B[N_TRANSIENT][N_CLASSES] = {0};
    for (int i = 0; i < N_TRANSIENT; i++) {
        for (int j = 0; j < N_CLASSES; j++) {
            for (int k = 0; k < N_TRANSIENT; k++) {
                B[i][j] += M[i][k] * R[k][j];
            }
        }
    }
    
    // Print Results
    int states[N_TRANSIENT] = {10, 19, 22, 24};
    printf("State | P(s -> A)  | P(s -> B)  | P(s -> C3)\n");
    printf("------|------------|------------|------------\n");
    for (int i = 0; i < N_TRANSIENT; i++) {
        printf("  %2d  |  %8.6f  |  %8.6f  |  %8.6f\n", 
               states[i], B[i][0], B[i][1], B[i][2]);
    }

    return 0;
}