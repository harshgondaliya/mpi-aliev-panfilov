/*
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include "cblock.h"
#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;

void printMat(const char mesg[], double *E, int m, int n);
void printMat3(const char mesg[], double *E, int m, int n);
double *alloc1D(int m,int n);

extern control_block cb;


//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n){
    int i;

    for (i = 0;i < (m + 2) * (n + 2);i++)   // including ghost cells
        E_prev[i] = R[i] = 0;

    for (i = (n + 2);i < (m + 1) * (n + 2);i++) {   // actual block, excluding ghost cells
        int colIndex = i % (n + 2);		// gives the base index (first row's) of the current index

        // Need to compute (n + 1) / 2 rather than n / 2 to work with odd numbers
        if(colIndex == 0 || colIndex == (n + 1) || colIndex < ((n + 1) / 2 + 1))
        continue;
        E_prev[i] = 1.0;
    }

    for (i = 0;i < (m + 2) * (n + 2);i++) {
        int rowIndex = i / (n + 2);		// gives the current row number in 2D array representation
        int colIndex = i % (n + 2);		// gives the base index (first row's) of the current index

        // Need to compute (m + 1) / 2 rather than m / 2 to work with odd numbers
        if(colIndex == 0 || colIndex == (n + 1) || rowIndex < ((m + 1) / 2 + 1))
        continue;
        R[i] = 1.0;
    }
// We only print the meshes if they are small enough
#if 0
    printMat("E_prev",E_prev, m, n);
    printMat("R",R, m, n);
#endif
#ifdef _MPI_
    int nprocs, myrank, n_rows, n_cols, rx, ry;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {
        for (int rank = nprocs - 1;rank >= 0;rank--) {
            rx = rank / cb.py;  // processor location x from m x n (row number of processor)
            ry = rank % cb.py;  // processor location y from m x n (col number of processor)

            int mod_x = (m + 2) % cb.px;
            int mod_y = (n + 2) % cb.py;

            n_rows = (m + 2) / cb.px;
            n_cols = (n + 2) / cb.py;

            if (mod_x != 0 && rx < mod_x) {
                n_rows += 1;
            }

            if (mod_y != 0 && ry < mod_y) {
                n_cols += 1;
            }

            int uniform_n_rows = (m + 2) / cb.px;
            int uniform_n_cols = (n + 2) / cb.py;
            int row_start = rx * uniform_n_rows;
            int col_start = ry * uniform_n_cols;

            if(mod_x != 0) {
                if(rx >= mod_x) {
                    row_start += mod_x;
                } else {
                    row_start += rx;
                }
            }
            if(mod_y != 0) {
                if(ry >= mod_y) {
                    col_start += mod_y;
                } else {
                    col_start += ry;
                }
            }
            // logic to add additional rows
            if(rx==0 && rx<cb.px-1){
                n_rows += 1;
            } else if(rx>0 && rx<cb.px-1){
                n_rows += 2;
                row_start -=1;
            } else if(rx==cb.px-1 && rx>0){
                n_rows += 1;
                row_start -=1;
            } // if -x = 1 then do not need to add any ghost rows.

            // logic to add additional columns
            if(ry==0 && ry<cb.py-1){
                n_cols += 1;
            } else if(ry>0 && ry<cb.py-1){
                n_cols += 2;
                col_start -=1;
            } else if(ry==cb.py-1 && ry>0){
                n_cols += 1;
                col_start -=1;
            } // if -y = 1 then do not need to add any ghost columns.

            double *submeshE_prev = alloc1D(n_rows, n_cols);
            double *submeshR = alloc1D(n_rows, n_cols);

            int row_end = row_start + n_rows;
            int col_end = col_start + n_cols;
            int index = 0;
            for (int row = row_start;row < row_end; row++) {
                for (int col = col_start; col < col_end; col++) {
//                    submeshE[index] = E[row * (n + 2) + col];
                    submeshE_prev[index] = E_prev[row * (n + 2) + col];
                    submeshR[index] = R[row * (n + 2) + col];
                    index++;
                }
            }
            if (rank != 0) {
                MPI_Request send_request[3];
                MPI_Status send_status[3];
//                MPI_Isend(submeshE, n_rows * n_cols, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &send_request[0]);
                MPI_Isend(submeshE_prev, n_rows * n_cols, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD, &send_request[1]);
                MPI_Isend(submeshR, n_rows * n_cols, MPI_DOUBLE, rank, 2, MPI_COMM_WORLD, &send_request[2]);

//                MPI_Wait(&send_request[0], &send_status[0]);
                MPI_Wait(&send_request[1], &send_status[1]);
                MPI_Wait(&send_request[2], &send_status[2]);
            } else {
                int index = 0;
                for (int i = 0;i < n_rows;i++)
                    for (int j = 0;j < n_cols;j++) {
//                        E[i * n_cols + j] = submeshE[i * n_cols + j];
                        E_prev[index] = submeshE_prev[i * n_cols + j];
                        R[index] = submeshR[i * n_cols + j];
                        index++;
                    }
            }
//            free(submeshE);
            free(submeshE_prev);
            free(submeshR);
        }
    } else {    // receive from source 0
        rx = myrank / cb.py;
        ry = myrank % cb.py;
        int mod_x = (m + 2) % cb.px;
        int mod_y = (n + 2) % cb.py;

        n_rows = (m + 2) / cb.px;
        n_cols = (n + 2) / cb.py;

        if (mod_x != 0 && rx < mod_x) {
            n_rows += 1;
        }

        if (mod_y != 0 && ry < mod_y) {
            n_cols += 1;
        }

        // logic to add additional rows
        if(rx==0 && rx<cb.px-1){
            n_rows += 1;
        } else if(rx>0 && rx<cb.px-1){
            n_rows += 2;
        } else if(rx==cb.px-1 && rx>0){
            n_rows += 1;
        } // if -x = 1 then do not need to add any ghost rows.

        // logic to add additional columns
        if(ry==0 && ry<cb.py-1){
            n_cols += 1;
        } else if(ry>0 && ry<cb.py-1){
            n_cols += 2;
        } else if(ry==cb.py-1 && ry>0){
            n_cols += 1;
        } // if -y = 1 then do not need to add any ghost columns.

        MPI_Request recv_request[3];
        MPI_Status recv_status[3];
        int source = 0;
//        MPI_Irecv(E, n_rows * n_cols, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &recv_request[1]);
        MPI_Irecv(E_prev, n_rows * n_cols, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &recv_request[1]);
        MPI_Irecv(R, n_rows * n_cols, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &recv_request[2]);

//        MPI_Wait(&recv_request[0], &recv_status[0]);
        MPI_Wait(&recv_request[1], &recv_status[1]);
        MPI_Wait(&recv_request[2], &recv_status[2]);
    }
#endif
}

double *alloc1D(int m,int n){
    int nx = n, ny = m;
    double *E;
    // Ensures that allocated memory is aligned on a 16 byte boundary
    assert(E = (double*)memalign(16, sizeof(double) * nx * ny));
    return(E);
}

void printMat(const char mesg[], double *E, int m, int n){
    int i;
    #if 0
    if (m>8)
        return;
    #else
    if (m>34)
        return;
    #endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
        int rowIndex = i / (n+2);
        int colIndex = i % (n+2);
        //      if ((colIndex>0) && (colIndex<n+1))
        //       if ((rowIndex > 0) && (rowIndex < m+1))
        printf("%6.3f ", E[i]);
        if (colIndex == n+1)
            printf("\n");
    }
}

void printMat3(const char mesg[], double *E, int m, int n){
    int i;
    #if 0
    if (m>8)
        return;
    #else
    if (m>34)
        return;
    #endif
    printf("%s\n",mesg);
    for (i = 0; i < m * n; i++) {
        printf("%6.3f ", E[i]);
        if (i % n == n - 1)
            printf("\n");
    }
}
