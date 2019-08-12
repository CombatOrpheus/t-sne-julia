#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "tsne.h"

// Function that runs the Barnes-Hut implementation of t-SNE
int main() {

    // Define some variables
	int origN, N, D, no_dims, max_iter, *landmarks;
	double perc_landmarks;
	double perplexity, theta, *data;
    int rand_seed = -1;
    TSNE* tsne = new TSNE();

    // Read the parameters and the dataset
	if(TSNE::load_data(&data, &origN, &D, &no_dims, &theta, &perplexity, &rand_seed, &max_iter)) {

		// Make dummy landmarks
        N = origN;
        int* landmarks = (int*) malloc(N * sizeof(int));
        if(landmarks == nullptr) { printf("Memory allocation failed!\n"); exit(1); }
        for(int n = 0; n < N; n++) landmarks[n] = n;

		// Now fire up the SNE implementation
		auto* Y = (double*) malloc(N * no_dims * sizeof(double));
		auto* costs = (double*) calloc(N, sizeof(double));
        if(Y == nullptr || costs == nullptr) { printf("Memory allocation failed!\n"); exit(1); }
		tsne->run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, false, max_iter);

		// Save the results
		TSNE::save_data(Y, landmarks, costs, N, no_dims);

        // Clean up the memory
		free(data); data = nullptr;
		free(Y); Y = nullptr;
		free(costs); costs = nullptr;
		free(landmarks); landmarks = nullptr;
    }
    delete(tsne);
}
