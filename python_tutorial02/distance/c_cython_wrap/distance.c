#include <math.h>

void calc_distance(int c1_size, int c2_size, double** c1, double** c2, double** distance) {
    int i, j;
    double dx, dy, dz;
    for (i = 0; i < c1_size; i++) {
        for (j = 0; j < c2_size; j++) {
            dx = c1[i][0] - c2[j][0];
            dy = c1[i][1] - c2[j][1];
            dz = c1[i][2] - c2[j][2];
            distance[i][j] = sqrt(dx*dx + dy*dy + dz*dz);
        }
    }
}
