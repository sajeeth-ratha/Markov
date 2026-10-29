
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

#define N MAX_SIZE
#define NUM_STEPS 10
#define MAX_SCENARIO_STATES 6

typedef struct {
    const char *name;
    int count;
    int states[MAX_SCENARIO_STATES];
    double probs[MAX_SCENARIO_STATES];
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
            fprintf(stderr,
                    "Error: invalid initial condition in %s.\n",
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

/* Print a table header with the selected state numbers as columns. */
void print_table_header(FILE *out, const int states[], int count)
{
    fprintf(out, "%-8s", "Step");

    for (int j = 0; j < count; j++) {
        char heading[32];
        snprintf(heading, sizeof(heading), "State %d", states[j]);
        fprintf(out, "%12s", heading);
    }

    fprintf(out, "%16s\n", "Total shown");

    for (int j = 0; j < 8 + 12 * count + 16; j++)
        fputc('-', out);

    fputc('\n', out);
}

/* Print one row of selected state probabilities. */
void print_distribution(FILE *out, const Matrix *V, int step,
                        const int states[], int count)
{
    double sum = 0.0;

    fprintf(out, "%-8d", step);

    for (int j = 0; j < count; j++) {
        double probability = V->data[0][states[j] - 1];
        fprintf(out, "%12.6f", probability);
        sum += probability;
    }

    fprintf(out, "%16.6f\n", sum);
}

/* Run one scenario and print only the requested states. */
void run_scenario(FILE *out, const Matrix *P,
                  int question, const Scenario *s)
{
    const int steps[NUM_STEPS] = {
        0, 1, 2, 10, 50, 51, 52, 100, 500, 1000
    };

    const int states_A[] = {2, 5, 12, 21, 25};
    const int states_B[] = {8, 9, 16};
    const int states_Q5[] = {10, 14, 19, 22, 24};
    const int states_C[] = {6, 17, 20};
    const int states_Q7[] = {3, 7, 23, 6, 17, 20};
    const int states_D[] = {1, 4, 11, 18, 26, 27};

    const int *states = NULL;
    int count = 0;

    switch (question) {
        case 1:
        case 2:
        case 3:
            states = states_A;
            count = 5;
            break;

        case 4:
            states = states_B;
            count = 3;
            break;

        case 5:
            states = states_Q5;
            count = 5;
            break;

        case 6:
            states = states_C;
            count = 3;
            break;

        case 7:
            states = states_Q7;
            count = 6;
            break;

        case 10:
            states = states_D;
            count = 6;
            break;

        default:
            fprintf(out, "Unknown question number %d\n", question);
            return;
    }

    Matrix V = make_initial(s);

    fprintf(out, "\n");
    for (int i = 0; i < 72; i++)
        fputc('=', out);

    fprintf(out, "\nQUESTION %d: %s\n", question, s->name);

    for (int i = 0; i < 72; i++)
        fputc('=', out);

    fprintf(out, "\n");
    print_table_header(out, states, count);
    print_distribution(out, &V, 0, states, count);

    int next_step = 1;

    for (int n = 1; n <= steps[NUM_STEPS - 1]; n++) {
        V = multiply_matrices(V, *P);

        double probability_sum = 0.0;
        for (int j = 0; j < N; j++)
            probability_sum += V.data[0][j];

        if (fabs(probability_sum - 1.0) > 1e-6) {
            fprintf(out,
                    "ERROR: probability sum is %.10f at step %d.\n",
                    probability_sum, n);
            return;
        }

        if (next_step < NUM_STEPS && n == steps[next_step]) {
            print_distribution(out, &V, n, states, count);
            next_step++;
        }
    }

    fprintf(out, "\n");
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

    const Scenario q1 = {
        "Start in State 2", 1, {2}, {1.0}
    };

    const Scenario q2 = {
        "Equal distribution on {2,5,12,21,25}",
        5, {2, 5, 12, 21, 25},
        {0.2, 0.2, 0.2, 0.2, 0.2}
    };

    const Scenario q3 = {
        "Arbitrary distribution on {2,5,12,21,25}",
        5, {2, 5, 12, 21, 25},
        {0.10, 0.30, 0.20, 0.25, 0.15}
    };

    const Scenario q4a = {
        "Start in State 8", 1, {8}, {1.0}
    };

    const Scenario q4b = {
        "Equal distribution on {8,9,16}",
        3, {8, 9, 16},
        {1.0/3.0, 1.0/3.0, 1.0/3.0}
    };

    const Scenario q4c = {
        "Arbitrary distribution on {8,9,16}",
        3, {8, 9, 16}, {0.50, 0.30, 0.20}
    };

    const Scenario q5a = {
        "Start in State 14", 1, {14}, {1.0}
    };

    const Scenario q5b = {
        "Equal distribution on {10,14,19,22,24}",
        5, {10, 14, 19, 22, 24},
        {0.2, 0.2, 0.2, 0.2, 0.2}
    };

    const Scenario q5c = {
        "Arbitrary distribution on {10,14,19,22,24}",
        5, {10, 14, 19, 22, 24},
        {0.10, 0.25, 0.30, 0.15, 0.20}
    };

    const Scenario q6a = {
        "Start in State 6", 1, {6}, {1.0}
    };

    const Scenario q6b = {
        "Equal distribution on {6,17,20}",
        3, {6, 17, 20},
        {1.0/3.0, 1.0/3.0, 1.0/3.0}
    };

    const Scenario q6c = {
        "Arbitrary distribution on {6,17,20}",
        3, {6, 17, 20}, {0.50, 0.30, 0.20}
    };

    const Scenario q7a = {
        "Start in State 3", 1, {3}, {1.0}
    };

    const Scenario q7b = {
        "Equal distribution on {3,7,23}",
        3, {3, 7, 23},
        {1.0/3.0, 1.0/3.0, 1.0/3.0}
    };

    const Scenario q7c = {
        "Arbitrary distribution on {3,7,23}",
        3, {3, 7, 23}, {0.50, 0.30, 0.20}
    };

    const Scenario q10a = {
        "Start in State 1", 1, {1}, {1.0}
    };

    const Scenario q10b = {
        "Equal distribution on Class D",
        6, {1, 4, 11, 18, 26, 27},
        {1.0/6.0, 1.0/6.0, 1.0/6.0,
         1.0/6.0, 1.0/6.0, 1.0/6.0}
    };

    const Scenario q10c = {
        "Arbitrary distribution on Class D",
        6, {1, 4, 11, 18, 26, 27},
        {0.50, 0.10, 0.10, 0.10, 0.10, 0.10}
    };

    const Scenario q10d = {
        "Stationary distribution of Class D",
        6, {1, 4, 11, 18, 26, 27},
        {5.0/18.0, 5.0/18.0,
         1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0}
    };

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

    run_scenario(out, &P, 10, &q10a);
    run_scenario(out, &P, 10, &q10b);
    run_scenario(out, &P, 10, &q10c);
    run_scenario(out, &P, 10, &q10d);

    fclose(out);

    printf("All simulations completed.\n");
    printf("Results saved in simulation_results.txt\n");

    return EXIT_SUCCESS;
}
