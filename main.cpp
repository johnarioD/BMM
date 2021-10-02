#include <vector>
#include <iostream>
#include <random>
#include <time.h>
#include <algorithm>
#include <mpi.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

using namespace std;

bool parse_args(int *N, int *M, int *D, int *nzA, int *nzB, int *nzF, int argc, char **argv);
void init_matrix(int **rows, int **cols, int N, int nz, bool generateRandoml, int maxNonZerosPerRowy);
void createSparse(int *rows, int *cols, int N, int nz, int maxNonZerosPerRow);
vector<int> serial(int *rowsA, int *colsA, int *rowsB, int *colsB, int *rowsC, int N);
vector<int> filterSerial(int *rowsA, int *colsA, int *rowsB, int *colsB, int *rowsF, int *colsF, int *rowsC, int N);
vector<int> BMM(int *rowsA, int *colsA, int *rowsB, int *colsB, int *rowsC, int N);
vector<int> filterBMM(int *rowsA, int *colsA, int *rowsB, int *colsB, int *rowsF, int *colsF, int *rowsC, int N);

int main(int argc, char **argv)
{
	int tid, tNum, err;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &tNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &tid);
	MPI_Request *mpireqs;
	int mx = (6 < tNum) ? tNum : 6;
	mpireqs = (MPI_Request *)malloc(mx * sizeof(MPI_Request));
	MPI_Status mpistat;

    // Matrices A, B, F
    // C = F.*(A*B)
    struct timespec ts_start, ts_end;
    int N, M, D;       // A = N*D  |  B = D*M | F = N*D
    int nzA, nzB, nzF; // amount of non-zeros for A,B,F
    int *rowsF, *colsF, *rowsA, *colsA, *rowsB, *colsB, *rowsC, *colsC;

    bool error = parse_args(&N, &M, &D, &nzA, &nzB, &nzF, argc, argv);

    if (error)
    {
        cout << "Missing arguments. Try using --help to find the correct format." << endl;
        MPI_Finalize();
        return 1;
    }

    int n = N/tNum + 1;
    int offset = tid*n;
    if(tid > N%tNum){
        offset -= (tid - N%tNum);
    }

    bool generateMatrixValues = false;
    if(tid==0){
        generateMatrixValues = true;
    }
    init_matrix(&rowsA, &colsA, N, nzA, generateMatrixValues, 4);
    init_matrix(&rowsB, &colsB, D, nzB, generateMatrixValues, 4);
    if (nzF > 0)
        init_matrix(&rowsF, &colsF, N, nzF, generateMatrixValues, 4);
    rowsC = (int *)calloc(n + 1, sizeof(int));
    vector<int> tmpColsC;

    if(tid==0){
    	clock_gettime(CLOCK_MONOTONIC, &ts_start);
    }    

	MPI_Bcast(rowsA, N, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rowsB, D, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rowsF, N, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(colsA, nzA, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(colsB, nzB, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(colsF, nzF, MPI_INT, 0, MPI_COMM_WORLD);

    if(tid >= N%tNum)
        n--;
    
    if (nzF > 0)
        tmpColsC = filterBMM(rowsA, colsA, rowsB, colsB, rowsF, colsF, rowsC, N);
    else
        tmpColsC = BMM(rowsA, colsA, rowsB, colsB, rowsC, N);
        
	colsC = (int *)malloc(tmpColsC.size() * sizeof(int));
	copy(tmpColsC.begin(), tmpColsC.end(), colsC);

	int resSize = tmpColsC.size();
	if (tid == 0)
	{
		int count = n;
		rowsC = (int *)realloc(rowsC, (N + 1) * sizeof(int));
		for (int i = 1; i < tNum; i++)
		{
			if (i == (N % tNum - 1))
			{
				n--;
			}
			cout << "Receiving rowsC[" << count << "-" << count + n - 1 << "] from " << i << endl;
			MPI_Irecv(rowsC + count, n, MPI_INT, i, 10 + i, MPI_COMM_WORLD, &mpireqs[i]);
			count += n;
		}
		for (int i = 1; i < tNum; i++)
		{
			MPI_Wait(&mpireqs[i], &mpistat);
		}
		cout << "Done this" << endl;

		int *tmp = (int *)malloc(tNum * sizeof(int));
		count = tmpColsC.size();
		for (int i = 1; i < tNum; i++)
		{
			MPI_Irecv(tmp + i, sizeof(int), MPI_INT, i, 10 + i, MPI_COMM_WORLD, &mpireqs[i]);
		}
		for (int i = 1; i < tNum; i++)
		{
			MPI_Wait(&mpireqs[i], &mpistat);
		}
		for (int i = 1; i < tNum; i++)
		{
			count += tmp[i];
		}

		cout << "New size should be " << count << endl;
		int *colsC = (int *)realloc(colsC, count * sizeof(int));
		count = resSize;
		for (int i = 1; i < tNum; i++)
		{
			MPI_Irecv(colsC + count, tmp[i], MPI_INT, i, 10 + i, MPI_COMM_WORLD, &mpireqs[i]);
			count += tmp[i];
		}
		for (int i = 1; i < tNum; i++)
		{
			MPI_Wait(&mpireqs[i], &mpistat);
		}
	}
	else
	{
		MPI_Send(rowsC, n, MPI_INT, 0, 10 + tid, MPI_COMM_WORLD);
		MPI_Send(&resSize, 1, MPI_INT, 0, 10 + tid, MPI_COMM_WORLD);
		MPI_Send(colsC, resSize, MPI_INT, 0, 10 + tid, MPI_COMM_WORLD);
	}

	if (tid == 0)
		clock_gettime(CLOCK_MONOTONIC, &ts_end);

	if (tid == 0)
		cout << "Duration : " << (ts_end.tv_sec - ts_start.tv_sec) << "." << abs(ts_end.tv_nsec - ts_start.tv_nsec) << endl;

	MPI_Finalize();

    return 0;
}

bool parse_args(int *N, int *M, int *D, int *nzA, int *nzB, int *nzF, int argc, char **argv)
{
    *nzA = 0;
    *N = 0;
    bool hasNZA = false;
    bool hasNZB = false;
    bool hasNZF = false;
    bool hasRowsA = false;
    bool hasRowsB = false;

    for (int i = 0; i < argc; i++)
    {
        string arg = argv[i];
        if (arg.compare("--nzA")==0 and (i + 1 < argc))
        {
            hasNZA = true;
            *nzA = atoi(argv[i+1]);
        }
        else if (arg.compare("--nzB")==0 and (i + 1 < argc))
        {
            hasNZB = true;
            *nzB = atoi(argv[i+1]);
        }
        else if (arg.compare("--nzF")==0 and (i + 1 < argc))
        {
            hasNZF = true;
            *nzF = atoi(argv[i+1]);
        }
        else if (arg.compare("--rA")==0 and (i + 1 < argc))
        {
            hasRowsA = true;
            *N = atoi(argv[i+1]);
        }
        else if (arg.compare("--rB")==0 and (i + 1 < argc))
        {
            hasRowsB = true;
            *D = atoi(argv[i+1]);
        }
        else if(arg.compare("--help")==0){
            cout << "\nThis program generates and multiplies sparse boolean matrices via the following operation:\n"
                 << "\n\tC = F .* ( A * B )\n"
                 << "\nArguments:\n\n"
                 << "* --nzA\t\tNumber of non zeros in matrix A\n"
                 << "  --nzB\t\tNumber of non zeros in matrix B\n"
                 << "  --nzF\t\tNumber of non zeros in matrix F\n"
                 << "* --rA\t\tNumber of rows in matrix A\n"
                 << "  --rB\t\tNUmber of rows in matrix B\n"
                 << "\n* always required\n"
                 << endl;
        }
    }

    if (!hasNZB)
    {
        *nzB = *nzA;
    }
    if (!hasNZF)
    {
        *nzF = 0;
    }
    if (!hasRowsB)
    {
        *D = *N;
    }

    bool error = !(hasNZA && hasRowsA);

    return error;
}

void init_matrix(int **rows, int **cols, int N, int nz, bool generate_randomly, int maxNonZerosPerRow)
{
    *rows = (int *)calloc(N + 1, sizeof(int));
    *cols = (int *)malloc(nz * sizeof(int));

    if (generate_randomly)
    {
        createSparse(*rows, *cols, N, nz, maxNonZerosPerRow);
    }
}

void createSparse(int *rows, int *cols, int N, int nz, int maxNonZerosPerRow)
{
    rows[0] = 0;
    for (int i = 1; i < N + 1; i++)
    {
        rows[i] = rows[i - 1] + rand() % maxNonZerosPerRow;
        if (rows[i] >= nz)
        {
            rows[i] = nz;
            for (int j = i + 1; j < N + 1; j++)
            {
                rows[j] = nz;
            }
            break;
        }
        else
        {
            for (int j = rows[i - 1]; j < rows[i]; j++)
            {
                cols[j] = rand() % N;
            }
        }
    }
}

vector<int> serial(int *rowsA, int *colsA, int *rowsB, int *colsB, int *rowsC, int N)
{
    vector<int> tmp;
    vector<int> resCols;

    for (int i = 0; i < N; i++)
    {
        for (int j = rowsA[i]; j < rowsA[i + 1]; j++)
        {
            for (int k = rowsB[colsA[j]]; k < rowsB[colsA[j] + 1]; k++)
            {
                if (!binary_search(tmp.begin(), tmp.end(), colsB[k]))
                {
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

vector<int> filterSerial(int *rowsA, int *colsA, int *rowsB, int *colsB, int *rowsF, int *colsF, int *rowsC, int N)
{
    vector<int> tmp;
    vector<int> resCols;

    for (int i = 0; i < N; i++)
    {
        for (int j = rowsA[i]; j < rowsA[i + 1]; j++)
        {
            int idxB = rowsB[colsA[j]];
            int idxF = rowsF[colsA[j]];
            while ((idxB < rowsB[colsA[j] + 1]) && (idxF < rowsF[colsA[j] + 1]))
            {
                if (colsB[idxB] < colsF[idxF])
                {
                    idxB++;
                }
                else if (colsB[idxB] > colsF[idxF])
                {
                    idxF++;
                }
                else
                {
                    if (!binary_search(tmp.begin(), tmp.end(), colsB[idxB]))
                    {
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

vector<int> BMM(int *rowsA, int *colsA, int *rowsB, int *colsB, int *rowsC, int N)
{
    vector<int> tmp;
    vector<int> resCols;

	__cilkrts_set_param("nworkers", "8");
    cilk_for(int i = 0; i < N; i++)
    {
        for (int j = rowsA[i]; j < rowsA[i + 1]; j++)
        {
            for (int k = rowsB[colsA[j]]; k < rowsB[colsA[j] + 1]; k++)
            {
                if (!binary_search(tmp.begin(), tmp.end(), colsB[k]))
                {
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

vector<int> filterBMM(int *rowsA, int *colsA, int *rowsB, int *colsB, int *rowsF, int *colsF, int *rowsC, int N)
{
    vector<int> tmp;
    vector<int> resCols;

	__cilkrts_set_param("nworkers", "8");
    cilk_for(int i = 0; i < N; i++)
    {
        for (int j = rowsA[i]; j < rowsA[i + 1]; j++)
        {
            int idxB = rowsB[colsA[j]];
            int idxF = rowsF[colsA[j]];
            while ((idxB < rowsB[colsA[j] + 1]) && (idxF < rowsF[colsA[j] + 1]))
            {
                if (colsB[idxB] < colsF[idxF])
                {
                    idxB++;
                }
                else if (colsB[idxB] > colsF[idxF])
                {
                    idxF++;
                }
                else
                {
                    if (!binary_search(tmp.begin(), tmp.end(), colsB[idxB]))
                    {
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
