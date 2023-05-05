/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <malloc.h>
#include <emmintrin.h>
#ifdef _MPI_
#include <mpi.h>
#endif
using namespace std;

#define E_LEFT_TAG 0
#define E_TOP_TAG 1
#define E_RIGHT_TAG 2
#define E_BOTTOM_TAG 3

#define R_LEFT_TAG 4
#define R_TOP_TAG 5
#define R_RIGHT_TAG 6
#define R_BOTTOM_TAG 7

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
void printMat4(const char mesg[], double *E, int m, int n);

extern control_block cb;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) (cb.m*cb.n);
    l2norm = sqrt(l2norm);
    return l2norm;
}
#ifndef _MPI_
void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;
 int m = cb.m, n=cb.n;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);


 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){
  
      if  (cb.debug && (niter==0)){
	  stats(E_prev,m,n,&mx,&sumSq);
          double l2norm = L2Norm(sumSq);
	  repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	  if (cb.plot_freq)
	      plotter->updatePlot(E,  -1, m+1, n+1);
      }

   /* 
    * Copy data from boundary of the computational box to the
    * padding region, set up for differencing computational box's boundary
    *
    * These are physical boundary conditions, and are not to be confused
    * with ghost cells that we would use in an MPI implementation
    *
    * The reason why we copy boundary conditions is to avoid
    * computing single sided differences at the boundaries
    * which increase the running time of solve()
    *
    */
    
    // 4 FOR LOOPS set up the padding needed for the boundary conditions
    int i,j;

    // Fills in the TOP Ghost Cells
    for (i = 0; i < (n+2); i++) {
        E_prev[i] = E_prev[i + (n+2)*2];
    }

    // Fills in the RIGHT Ghost Cells
    for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i-2];
    }

    // Fills in the LEFT Ghost Cells
    for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i+2];
    }	

    // Fills in the BOTTOM Ghost Cells
    for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
        E_prev[i] = E_prev[i - (n+2)*2];
    }

//////////////////////////////////////////////////////////////////////////////

#define FUSED 1

#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
	E_prev_tmp = E_prev + j;
        R_tmp = R + j;
	for(i = 0; i < n; i++) {
	    E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
	E_prev_tmp = E_prev + j;
        for(i = 0; i < n; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
     /////////////////////////////////////////////////////////////////////////////////

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	    plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

  stats(E_prev,m,n,&Linf,&sumSq);
  L2 = L2Norm(sumSq);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}
#else
void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

    // Simulated time is different from the integer timestep number
    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;
    int m = cb.m, n=cb.n;
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int rx = myrank / cb.py;  // processor location x from m x n (row number of processor)
    int ry = myrank % cb.py;  // processor location y from m x n (col number of processor)

    int mod_x = (cb.m + 2) % cb.px;
    int mod_y = (cb.n + 2) % cb.py;

    int n_rows = (cb.m + 2) / cb.px;
    int n_cols = (cb.n + 2) / cb.py;

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
    // if(myrank == 0){
    //     printf("myrank=%d, n_cols=%d, n_rows=%d\n",myrank, n_cols, n_rows);
    // }
    int innerBlockRowStartIndex = n_cols+1;
    int innerBlockRowEndIndex = (n_rows*n_cols-1) - 2*n_cols + 2;

    MPI_Datatype row_type, col_type;	
    MPI_Type_contiguous	(n_cols,MPI_DOUBLE, &row_type);
    MPI_Type_vector(n_rows,1,n_cols,MPI_DOUBLE,&col_type);	
    MPI_Type_commit(&row_type);
    MPI_Type_commit(&col_type);

    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations
    for (niter = 0; niter < cb.niters; niter++){

        if  (cb.debug && (niter==0)){
            stats(E_prev,m,n,&mx,&sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
            if (cb.plot_freq)
                plotter->updatePlot(E,  -1, m+1, n+1);
        }

        /*
         * Copy data from boundary of the computational box to the
         * padding region, set up for differencing computational box's boundary
         *
         * These are physical boundary conditions, and are not to be confused
         * with ghost cells that we would use in an MPI implementation
         *
         * The reason why we copy boundary conditions is to avoid
         * computing single sided differences at the boundaries
         * which increase the running time of solve()
         *
         */

        // 4 FOR LOOPS set up the padding needed for the boundary conditions
        int i,j;
        MPI_Request sendEprev_request[4];
        MPI_Status sendEprev_status[4];
        MPI_Request sendR_request[4];
        MPI_Status sendR_status[4];
        int sent = 0;
        // complete all the send operations
        if(rx==0 && rx<cb.px-1){
            // Fills in the TOP Boundary Cells
            for (i = 0; i < n_cols; i++) {
                E_prev[i] = E_prev[i + n_cols*2];
            }
            MPI_Isend((E_prev + (n_rows-2)*n_cols), 1, row_type, (myrank + cb.py), E_BOTTOM_TAG, MPI_COMM_WORLD, &sendEprev_request[sent]);
            MPI_Isend((R + (n_rows-2)*n_cols), 1, row_type, (myrank + cb.py), R_BOTTOM_TAG, MPI_COMM_WORLD, &sendR_request[sent]);
            sent += 1;
        } else if(rx>0 && rx<cb.px-1){
            MPI_Isend((E_prev + (n_rows-2)*n_cols), 1, row_type, (myrank + cb.py), E_BOTTOM_TAG, MPI_COMM_WORLD, &sendEprev_request[sent]);
            MPI_Isend((R + (n_rows-2)*n_cols), 1, row_type, (myrank + cb.py), R_BOTTOM_TAG, MPI_COMM_WORLD, &sendR_request[sent]);
            sent += 1;

            MPI_Isend(E_prev + n_cols, 1, row_type, (myrank - cb.py), E_TOP_TAG, MPI_COMM_WORLD, &sendEprev_request[sent]);
            MPI_Isend(R + n_cols, 1, row_type, (myrank - cb.py), R_TOP_TAG, MPI_COMM_WORLD, &sendR_request[sent]);
            sent += 1;
        } else if(rx==cb.px-1 && rx>0){
            // Fills in the BOTTOM Boundary Cells
            for (i = (n_rows*n_cols-(n_cols)); i < n_rows*n_cols; i++) {
                E_prev[i] = E_prev[i - n_cols*2];
            }
            MPI_Isend(E_prev + n_cols, 1, row_type, (myrank - cb.py), E_TOP_TAG, MPI_COMM_WORLD, &sendEprev_request[sent]);
            MPI_Isend(R + n_cols, 1, row_type, (myrank - cb.py), R_TOP_TAG, MPI_COMM_WORLD, &sendR_request[sent]);
            sent += 1;
        } else if(rx==0 && rx==cb.px-1){ // handles case where -x = 1
            // Fills in the TOP Boundary Cells
            for (i = 0; i < n_cols; i++) {
                E_prev[i] = E_prev[i + n_cols*2];
            }
            // Fills in the BOTTOM Boundary Cells
            for (i = (n_rows*n_cols-(n_cols)); i < n_rows*n_cols; i++) {
                E_prev[i] = E_prev[i - n_cols*2];
            }
        }
        if(ry==0 && ry<cb.py-1){
            // Fills in the LEFT Boundary Cells
            for (i = 0; i < n_rows*n_cols; i+=n_cols) {
                E_prev[i] = E_prev[i+2];
            }
            MPI_Isend((E_prev + n_cols - 2), 1, col_type, (myrank + 1), E_RIGHT_TAG, MPI_COMM_WORLD, &sendEprev_request[sent]);
            MPI_Isend((R + n_cols - 2), 1, col_type, (myrank + 1), R_RIGHT_TAG, MPI_COMM_WORLD, &sendR_request[sent]);
            sent += 1;
        } else if(ry>0 && ry<cb.py-1){
            MPI_Isend((E_prev + n_cols - 2), 1, col_type, (myrank + 1), E_RIGHT_TAG, MPI_COMM_WORLD, &sendEprev_request[sent]);
            MPI_Isend((R + n_cols - 2), 1, col_type, (myrank + 1), R_RIGHT_TAG, MPI_COMM_WORLD, &sendR_request[sent]);
            sent += 1;

            MPI_Isend(E_prev+1, 1, col_type, (myrank - 1), E_LEFT_TAG, MPI_COMM_WORLD, &sendEprev_request[sent]);
            MPI_Isend(R+1, 1, col_type, (myrank - 1), R_LEFT_TAG, MPI_COMM_WORLD, &sendR_request[sent]);
            sent += 1;
        } else if(ry==cb.py-1 && ry>0){
            // Fills in the RIGHT Boundary Cells
            for (i = n_cols-1; i < n_rows*n_cols; i+=n_cols) {
                E_prev[i] = E_prev[i-2];
            }
            MPI_Isend(E_prev+1, 1, col_type, (myrank - 1), E_LEFT_TAG, MPI_COMM_WORLD, &sendEprev_request[sent]);
            MPI_Isend(R+1, 1, col_type, (myrank - 1), R_LEFT_TAG, MPI_COMM_WORLD, &sendR_request[sent]);
            sent += 1;
        } else if(ry==0 && ry==cb.py-1){ // handles case where -y = 1
            // Fills in the RIGHT Boundary Cells
            for (i = n_cols-1; i < n_rows*n_cols; i+=n_cols) {
                E_prev[i] = E_prev[i-2];
            }

            // Fills in the LEFT Boundary Cells
            for (i = 0; i < n_rows*n_cols; i+=n_cols) {
                E_prev[i] = E_prev[i+2];
            }
        }
        for(int i=0; i<sent; i++){
            MPI_Wait(&sendEprev_request[i], &sendEprev_status[i]);
            MPI_Wait(&sendR_request[i], &sendR_status[i]);
        }

    
        MPI_Request receiveEprev_request[4];
        MPI_Status receiveEprev_status[4];
        MPI_Request receiveR_request[4];
        MPI_Status receiveR_status[4];
        int receive = 0;

        // complete all the receive operations
        if(rx==0 && rx<cb.px-1){
            MPI_Irecv((E_prev + (n_rows-1)*n_cols), 1, row_type, (myrank + cb.py), E_TOP_TAG, MPI_COMM_WORLD, &receiveEprev_request[receive]);
            MPI_Irecv((R + (n_rows-1)*n_cols), 1, row_type, (myrank + cb.py), R_TOP_TAG, MPI_COMM_WORLD, &receiveR_request[receive]);
            receive += 1;
        } else if(rx>0 && rx<cb.px-1){
            MPI_Irecv((E_prev + (n_rows-1)*n_cols), 1, row_type, (myrank + cb.py), E_TOP_TAG, MPI_COMM_WORLD, &receiveEprev_request[receive]);
            MPI_Irecv((R + (n_rows-1)*n_cols), 1, row_type, (myrank + cb.py), R_TOP_TAG, MPI_COMM_WORLD, &receiveR_request[receive]);
            receive += 1;

            MPI_Irecv(E_prev, 1, row_type, (myrank - cb.py), E_BOTTOM_TAG, MPI_COMM_WORLD, &receiveEprev_request[receive]);
            MPI_Irecv(R, 1, row_type, (myrank - cb.py), R_BOTTOM_TAG, MPI_COMM_WORLD, &receiveR_request[receive]);
            receive += 1;
        } else if(rx==cb.px-1 && rx>0){
            MPI_Irecv(E_prev, 1, row_type, (myrank - cb.py), E_BOTTOM_TAG, MPI_COMM_WORLD, &receiveEprev_request[receive]);
            MPI_Irecv(R, 1, row_type, (myrank - cb.py), R_BOTTOM_TAG, MPI_COMM_WORLD, &receiveR_request[receive]);
            receive += 1;
        } 
        if(ry==0 && ry<cb.py-1){
            MPI_Irecv((E_prev + n_cols - 1), 1, col_type, (myrank + 1), E_LEFT_TAG, MPI_COMM_WORLD, &receiveEprev_request[receive]);
            MPI_Irecv((R + n_cols - 1), 1, col_type, (myrank + 1), R_LEFT_TAG, MPI_COMM_WORLD, &receiveR_request[receive]);
            receive += 1;
        } else if(ry>0 && ry<cb.py-1){
            MPI_Irecv((E_prev + n_cols - 1), 1, col_type, (myrank + 1), E_LEFT_TAG, MPI_COMM_WORLD, &receiveEprev_request[receive]);
            MPI_Irecv((R + n_cols - 1), 1, col_type, (myrank + 1), R_LEFT_TAG, MPI_COMM_WORLD, &receiveR_request[receive]);
            receive += 1;

            MPI_Irecv(E_prev, 1, col_type, (myrank - 1), E_RIGHT_TAG, MPI_COMM_WORLD, &receiveEprev_request[receive]);
            MPI_Irecv(R, 1, col_type, (myrank - 1), R_RIGHT_TAG, MPI_COMM_WORLD, &receiveR_request[receive]);
            receive += 1;
        } else if(ry==cb.py-1 && ry>0){
            // printf("myrank=%d\n",myrank);
            // MPI_Irecv(receivedColumn, n_rows, MPI_DOUBLE, (myrank - 1), E_RIGHT_TAG, MPI_COMM_WORLD, &receiveEprev_request[receive]);
            MPI_Irecv(E_prev, 1, col_type, (myrank - 1), E_RIGHT_TAG, MPI_COMM_WORLD, &receiveEprev_request[receive]);
            MPI_Irecv(R, 1, col_type, (myrank - 1), R_RIGHT_TAG, MPI_COMM_WORLD, &receiveR_request[receive]);
            receive += 1;
        }
        for(int i=0; i<receive; i++){
            MPI_Wait(&receiveEprev_request[i], &receiveEprev_status[i]);
            MPI_Wait(&receiveR_request[i], &receiveR_status[i]);
        }
        
//////////////////////////////////////////////////////////////////////////////

#define FUSED 1

#ifdef FUSED
        // Solve for the excitation, a PDE
        for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=n_cols) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
            for(i = 0; i < n_cols-2; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+n_cols]+E_prev_tmp[i-n_cols]);
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            }
        }
#else
        // Solve for the excitation, a PDE
        for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=n_cols) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n_cols-2; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+n_cols]+E_prev_tmp[i-n_cols]);
            }
        }

        /*
        * Solve the ODE, advancing excitation and recovery variables
        *     to the next timtestep
        */

        for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=n_cols) {
            E_tmp = E + j;
            R_tmp = R + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n_cols-2; i++) {
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            }
        }
#endif
        /////////////////////////////////////////////////////////////////////////////////

        if (cb.stats_freq){
            if ( !(niter % cb.stats_freq)){
                stats(E,m,n,&mx,&sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
            }
        }

        if (cb.plot_freq){
            if (!(niter % cb.plot_freq)){
                plotter->updatePlot(E,  niter, m, n);
            }
        }

        // Swap current and previous meshes
        double *tmp = E; E = E_prev; E_prev = tmp;

    } //end of 'niter' loop at the beginning

    //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

    stats(E_prev,n_rows-2,n_cols-2,&Linf,&sumSq);
    double finalSumSq, finalLinf;
    MPI_Reduce(&sumSq, &finalSumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Linf, &finalLinf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    Linf = finalLinf;
    L2 = L2Norm(finalSumSq);    

    *_E = E;
    *_E_prev = E_prev;
}
#endif
void printMat2(const char mesg[], double *E, int m, int n){
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
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}

void printMat4(const char mesg[], double *E, int m, int n){
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
