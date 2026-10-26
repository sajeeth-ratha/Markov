
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

#define N MAX_SIZE
#define NUM_STEPS 8

typedef struct {
    const char *name;
    int count;
    int states[5];
    double probs[5];
} Scenario;

/* Read and validate the transition matrix. */
int read_transition_matrix(const char *filename, Matrix *P)
{
    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        return 0;
    }

    P->rows = N;
    P->cols = N;

    for (int i = 0; i < N; i++) {
        double sum = 0.0;

        for (int j = 0; j < N; j++) {
            double x;

            if (fscanf(file, "%lf", &x) != 1) {
                fprintf(stderr,
                        "Error: expected %d x %d numbers.\n", N, N);
                fclose(file);
                return 0;
            }

            if (!isfinite(x) || x < 0.0 || x > 1.0) {
                fprintf(stderr,
                        "Error: invalid probability at row %d, column %d.\n",
                        i + 1, j + 1);
                fclose(file);
                return 0;
            }

            P->data[i][j] = x;
            sum += x;
        }

        if (fabs(sum - 1.0) > 1e-6) {
            fprintf(stderr,
                    "Error: row %d sums to %.10f, not 1.\n",
                    i + 1, sum);
            fclose(file);
            return 0;
        }
    }

    double extra;
    if (fscanf(file, "%lf", &extra) == 1) {
        fprintf(stderr, "Error: matrix contains extra numbers.\n");
        fclose(file);
        return 0;
    }

    fclose(file);
    return 1;
}

/* Build a row vector from a scenario. */
Matrix make_initial(const Scenario *s)
{
    Matrix V = create_matrix(1, N);

    for (int j = 0; j < N; j++)
        V.data[0][j] = 0.0;

    double sum = 0.0;

    for (int k = 0; k < s->count; k++) {
        int state = s->states[k];
        double probability = s->probs[k];

        if (state < 1 || state > N ||
            !isfinite(probability) ||
            probability < 0.0 || probability > 1.0) {
            fprintf(stderr, "Error: invalid initial condition in %s.\n",
                    s->name);
            exit(EXIT_FAILURE);
        }

        V.data[0][state - 1] += probability;
        sum += probability;
    }

    if (fabs(sum - 1.0) > 1e-6) {
        fprintf(stderr,
                "Error: initial probabilities for %s sum to %.10f.\n",
                s->name, sum);
        exit(EXIT_FAILURE);
    }

    return V;
}

/* Print the full distribution and its probability sum. */
void print_distribution(FILE *out, const Matrix *V, int step)
{
    double sum = 0.0;

    fprintf(out, "\nStep n = %d\n", step);

    for (int j = 0; j < N; j++) {
        fprintf(out, "State %2d: %.6f\n",
                j + 1, V->data[0][j]);
        sum += V->data[0][j];
    }

    fprintf(out, "Probability sum: %.10f\n", sum);

    if (fabs(sum - 1.0) > 1e-6)
        fprintf(out, "WARNING: probabilities do not sum to 1!\n");
}

/* Run one scenario at the selected steps. */
void run_scenario(FILE *out, const Matrix *P,
                  int question, const Scenario *s)
{
    const int steps[NUM_STEPS] = {
        0, 1, 2, 10, 50, 100, 500, 1000
    };

    Matrix V = make_initial(s);

    fprintf(out, "\n========================================\n");
    fprintf(out, "Question %d: %s\n", question, s->name);
    fprintf(out, "========================================\n");

    print_distribution(out, &V, 0);

    int next_step = 1;

    for (int n = 1; n <= steps[NUM_STEPS - 1]; n++) {
        V = multiply_matrices(V, *P);

        double sum = 0.0;
        for (int j = 0; j < N; j++)
            sum += V.data[0][j];

        if (fabs(sum - 1.0) > 1e-6) {
            fprintf(out,
                    "ERROR: probability sum is %.10f at step %d.\n",
                    sum, n);
            return;
        }

        if (n == steps[next_step]) {
            print_distribution(out, &V, n);
            next_step++;

            if (next_step == NUM_STEPS)
                break;
        }
    }
}

int main(void)
{
    Matrix P = create_matrix(N, N);

    if (!read_transition_matrix("matrix.txt", &P))
        return EXIT_FAILURE;

    FILE *out = fopen("simulation_results.txt", "w");

    if (out == NULL) {
        fprintf(stderr,
                "Error: cannot create simulation_results.txt\n");
        return EXIT_FAILURE;
    }

    /*
     * Questions 1-3:
     * Each question uses a different initial distribution
     * on the states {2, 5, 12, 21, 25}.
     */

    const Scenario q1 = {
        "Start in State 2",
        1,
        {2},
        {1.0}
    };

    const Scenario q2 = {
        "Equal distribution on {2,5,12,21,25}",
        5,
        {2, 5, 12, 21, 25},
        {0.2, 0.2, 0.2, 0.2, 0.2}
    };

    const Scenario q3 = {
        "Arbitrary distribution on {2,5,12,21,25}",
        5,
        {2, 5, 12, 21, 25},
        {0.10, 0.30, 0.20, 0.25, 0.15}
    };

    /*
     * Question 4: states {8, 9, 16}.
     */

    const Scenario q4a = {
        "Start in State 8",
        1,
        {8},
        {1.0}
    };

    const Scenario q4b = {
        "Equal distribution on {8,9,16}",
        3,
        {8, 9, 16},
        {1.0/3.0, 1.0/3.0, 1.0/3.0}
    };

    const Scenario q4c = {
        "Arbitrary distribution on {8,9,16}",
        3,
        {8, 9, 16},
        {0.50, 0.30, 0.20}
    };

    /*
     * Question 5: states {10, 14, 19, 22, 24}.
     */

    const Scenario q5a = {
        "Start in State 14",
        1,
        {14},
        {1.0}
    };

    const Scenario q5b = {
        "Equal distribution on {10,14,19,22,24}",
        5,
        {10, 14, 19, 22, 24},
        {0.2, 0.2, 0.2, 0.2, 0.2}
    };

    const Scenario q5c = {
        "Arbitrary distribution on {10,14,19,22,24}",
        5,
        {10, 14, 19, 22, 24},
        {0.10, 0.25, 0.30, 0.15, 0.20}
    };

    /*
     * Question 6: states {6, 17, 20}.
     */

    const Scenario q6a = {
        "Start in State 6",
        1,
        {6},
        {1.0}
    };

    const Scenario q6b = {
        "Equal distribution on {6,17,20}",
        3,
        {6, 17, 20},
        {1.0/3.0, 1.0/3.0, 1.0/3.0}
    };

    const Scenario q6c = {
        "Arbitrary distribution on {6,17,20}",
        3,
        {6, 17, 20},
        {0.50, 0.30, 0.20}
    };

    /*
     * Question 7: states {3, 7, 23}.
     */

    const Scenario q7a = {
        "Start in State 3",
        1,
        {3},
        {1.0}
    };

    const Scenario q7b = {
        "Equal distribution on {3,7,23}",
        3,
        {3, 7, 23},
        {1.0/3.0, 1.0/3.0, 1.0/3.0}
    };

    const Scenario q7c = {
        "Arbitrary distribution on {3,7,23}",
        3,
        {3, 7, 23},
        {0.50, 0.30, 0.20}
    };

    /* Execute all scenarios. */
    run_scenario(out, &P, 1, &q1);
    run_scenario(out, &P, 2, &q2);
    run_scenario(out, &P, 3, &q3);

    run_scenario(out, &P, 4, &q4a);
    run_scenario(out, &P, 4, &q4b);
    run_scenario(out, &P, 4, &q4c);

    run_scenario(out, &P, 5, &q5a);
    run_scenario(out, &P, 5, &q5b);
    run_scenario(out, &P, 5, &q5c);

    run_scenario(out, &P, 6, &q6a);
    run_scenario(out, &P, 6, &q6b);
    run_scenario(out, &P, 6, &q6c);

    run_scenario(out, &P, 7, &q7a);
    run_scenario(out, &P, 7, &q7b);
    run_scenario(out, &P, 7, &q7c);

    fclose(out);

    printf("All simulations completed.\n");
    printf("Results saved in simulation_results.txt\n");

    return EXIT_SUCCESS;
}
