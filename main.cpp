#include <vector>
#include <iostream>
#include <random>
#include <time.h>
#include <algorithm>
#include <mpi.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

using namespace std;

void createSparce(int *rows, int *cols, int N, int nz);
vector<int> serial(int* rowsA, int* colsA, int* rowsB, int* colsB, int* rowsC, int N);
vector<int> filterSerial(int* rowsA, int* colsA, int* rowsB, int* colsB, int* rowsF, int* colsF, int* rowsC, int N);
vector<int> BMM(int* rowsA, int* colsA, int* rowsB, int* colsB, int* rowsC, int N);
vector<int> filterBMM(int* rowsA, int* colsA, int* rowsB, int* colsB, int* rowsF, int* colsF, int* rowsC, int N);

int main(int argc, char** argv) {

	int tid, tNum, err;
  MPI_Init(&argc,&argv);
  MPI_Comm_size( MPI_COMM_WORLD, &tNum);
  MPI_Comm_rank( MPI_COMM_WORLD, &tid);
  MPI_Request * mpireqs;
	int mx = (6 < tNum) ? tNum : 6;
	mpireqs = (MPI_Request *)malloc(mx * sizeof(MPI_Request));
  MPI_Status mpistat;

	if(argc < 4){
		cout << "Invalid argument count" << endl;
		MPI_Finalize();
		return 0;
	}

	struct timespec ts_start, ts_end;
	int N, M, D, nz, nzF;
	int* rowsF, * colsF, * rowsA, * colsA, * rowsB, * colsB, * rowsC, * colsC;

	vector<int> tmpColsC;

	N = atoi(argv[1]);
	M = N;
	D = N;
	nz = atoi(argv[2]);
	nzF = atoi(argv[3]);

	int n = N/tNum + 1;
	int offset = tid*n;
	if(tid > N%tNum){
		 offset -= (tid - N%tNum);
	}

	rowsA = (int*)calloc(N + 1, sizeof(int));
	rowsB = (int*)calloc(D + 1, sizeof(int));
	rowsF = (int*)calloc(N + 1, sizeof(int));
	rowsC = (int*)calloc(n + 1, sizeof(int));

	colsA = (int*)malloc(nz * sizeof(int));
	colsB = (int*)malloc(nz * sizeof(int));
	colsF = (int*)malloc(nzF * sizeof(int));

	if(tid == 0){
		srand(time(NULL));
		createSparce(rowsA, colsA, N, nz);
		cout << "A full" << endl;
		createSparce(rowsB, colsB, D, nz);
		cout << "B full" << endl;
	}

	if(tid == 0)	clock_gettime(CLOCK_MONOTONIC, &ts_start);

	MPI_Bcast(rowsA, N, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rowsB, D, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rowsF, N, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(colsA, nz, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(colsB, nz, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(colsF, nzF, MPI_INT, 0, MPI_COMM_WORLD);

	if(tid >= N%tNum)	n--;
	//cout << "N is " << n << " for tid " << tid << endl;

	tmpColsC = BMM(rowsA + offset, colsA, rowsB, colsB, rowsC, n);
	colsC = (int*)malloc(tmpColsC.size() * sizeof(int));
	copy(tmpColsC.begin(), tmpColsC.end(), colsC);

	int resSize = tmpColsC.size();
	if(tid == 0){
		int count = n;
		rowsC = (int *)realloc(rowsC, (N+1)*sizeof(int));
		for(int i = 1; i < tNum; i++){
			if(i == (N%tNum -1)){
				n--;
			}
			cout << "Receiving rowsC[" << count << "-" << count+n-1 << "] from " << i << endl;
			MPI_Irecv(rowsC + count, n, MPI_INT, i, 10 + i, MPI_COMM_WORLD, &mpireqs[i]);
			count += n;
		}
		for(int i = 1; i < tNum; i++){
			MPI_Wait(&mpireqs[i],&mpistat);
		}
		cout << "Done this" << endl;

		int *tmp = (int *)malloc(tNum*sizeof(int));
		count = tmpColsC.size();
		for(int i = 1; i < tNum; i++){
			MPI_Irecv(tmp + i, sizeof(int), MPI_INT, i , 10+i, MPI_COMM_WORLD, &mpireqs[i]);
		}
		for(int i = 1; i < tNum; i++){
			MPI_Wait(&mpireqs[i],&mpistat);
		}
		for(int i = 1; i < tNum; i++){
			count += tmp[i];
		}

		cout << "New size should be " << count << endl;
		int * colsC = (int *)realloc(colsC, count*sizeof(int));
		count = resSize;
		for(int i = 1; i < tNum; i++){
			MPI_Irecv(colsC + count, tmp[i], MPI_INT, i, 10+i, MPI_COMM_WORLD, &mpireqs[i]);
			count += tmp[i];
		}
		for(int i = 1; i < tNum; i++){
			MPI_Wait(&mpireqs[i],&mpistat);
		}
	}else{
		MPI_Send(rowsC, n, MPI_INT, 0, 10+tid, MPI_COMM_WORLD);
		MPI_Send(&resSize, 1, MPI_INT, 0, 10+tid, MPI_COMM_WORLD);
		MPI_Send(colsC, resSize, MPI_INT, 0, 10+tid, MPI_COMM_WORLD);
	}

	if(tid == 0) clock_gettime(CLOCK_MONOTONIC, &ts_end);

	if(tid == 0) cout << "Duration : " << (ts_end.tv_sec - ts_start.tv_sec) << "." << abs(ts_end.tv_nsec - ts_start.tv_nsec) << endl;

	/*if (N < 6) {
		cout << "\nA:";
		for (int i = 0; i < N; i++) {
			cout << "\n|  ";
			for (int j = rowsA[i]; j < rowsA[i + 1]; j++) {
				cout << colsA[j] + 1 << "\t";
			}
		}
		cout << "\n\nB:";
		for (int i = 0; i < d; i++) {
			cout << "\n|  ";
			for (int j = rowsB[i]; j < rowsB[i + 1]; j++) {
				cout << colsB[j] + 1 << "\t";
			}
		}
		cout << "\n\nC:";
		for (int i = 0; i < N; i++) {
			cout << "\n|  ";
			for (int j = rowsC[i]; j < rowsC[i + 1]; j++) {
				cout << colsC[j] + 1 << "\t";
			}
		}
	}*/

	MPI_Finalize();

	return 0;
}

void createSparce(int *rows, int *cols, int N, int nz){
	int lim = 4;

	for (int i = 0; i < N; i++) {
		rows[i + 1] += rand() % lim;
		if (rows[i + 1] >= nz) {
			rows[i + 1] = nz;
			break;
		}
		for (int j = rows[i]; j < rows[i + 1]; j++) {
			cols[j] = rand() % N;
		}
	}
}

vector<int> serial(int* rowsA, int* colsA, int* rowsB, int* colsB, int* rowsC, int N) {
	vector<int> tmp;
	vector<int> resCols;

	for (int i = 0; i < N; i++) {
		for (int j = rowsA[i]; j < rowsA[i + 1]; j++) {
			for (int k = rowsB[colsA[j]]; k < rowsB[colsA[j] + 1]; k++) {
				if (!binary_search(tmp.begin(), tmp.end(), colsB[k])) {
					tmp.push_back(colsB[k]);
					sort(tmp.begin(), tmp.end());
				}
			}
		}
		resCols.insert(resCols.end(), tmp.begin(), tmp.end());
		rowsC[i + 1] = resCols.size();
		tmp.clear();
	}
	return resCols;
}

vector<int> filterSerial(int* rowsA, int* colsA, int* rowsB, int* colsB, int* rowsF, int* colsF, int* rowsC, int N) {
	vector<int> tmp;
	vector<int> resCols;

	for (int i = 0; i < N; i++) {
		for (int j = rowsA[i]; j < rowsA[i + 1]; j++) {
			int idxB = rowsB[colsA[j]];
			int idxF = rowsF[colsA[j]];
			while ((idxB < rowsB[colsA[j] + 1]) && (idxF < rowsF[colsA[j] + 1])) {
				if (colsB[idxB] < colsF[idxF]) {
					idxB++;
				}
				else if (colsB[idxB] > colsF[idxF]) {
					idxF++;
				}
				else {
					if (!binary_search(tmp.begin(), tmp.end(), colsB[idxB])) {
						tmp.push_back(colsB[idxB]);
						sort(tmp.begin(), tmp.end());
					}
					idxB++;
					idxF++;
				}
			}
		}
		resCols.insert(resCols.end(), tmp.begin(), tmp.end());
		rowsC[i + 1] = resCols.size();
		tmp.clear();
	}
	return resCols;
}

vector<int> BMM(int* rowsA, int* colsA, int* rowsB, int* colsB, int* rowsC, int N) {
	vector<int> tmp;
	vector<int> resCols;

	__cilkrts_set_param("nworkers","8");
	cilk_for (int i = 0; i < N; i++) {
		for (int j = rowsA[i]; j < rowsA[i + 1]; j++) {
			for (int k = rowsB[colsA[j]]; k < rowsB[colsA[j] + 1]; k++) {
				if (!binary_search(tmp.begin(), tmp.end(), colsB[k])) {
					tmp.push_back(colsB[k]);
					sort(tmp.begin(), tmp.end());
				}
			}
		}
		resCols.insert(resCols.end(), tmp.begin(), tmp.end());
		rowsC[i + 1] = resCols.size();
		tmp.clear();
	}
	return resCols;
}

vector<int> filterBMM(int* rowsA, int* colsA, int* rowsB, int* colsB, int* rowsF, int* colsF, int* rowsC, int N) {
	vector<int> tmp;
	vector<int> resCols;

	__cilkrts_set_param("nworkers","8");
	cilk_for (int i = 0; i < N; i++) {
		for (int j = rowsA[i]; j < rowsA[i + 1]; j++) {
			int idxB = rowsB[colsA[j]];
			int idxF = rowsF[colsA[j]];
			while ((idxB < rowsB[colsA[j] + 1]) && (idxF < rowsF[colsA[j] + 1])) {
				if (colsB[idxB] < colsF[idxF]) {
					idxB++;
				}
				else if (colsB[idxB] > colsF[idxF]) {
					idxF++;
				}
				else {
					if (!binary_search(tmp.begin(), tmp.end(), colsB[idxB])) {
						tmp.push_back(colsB[idxB]);
						sort(tmp.begin(), tmp.end());
					}
					idxB++;
					idxF++;
				}
			}
		}
		resCols.insert(resCols.end(), tmp.begin(), tmp.end());
		rowsC[i + 1] = resCols.size();
		tmp.clear();
	}
	return resCols;
}
