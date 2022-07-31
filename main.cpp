#include<stdlib.h>

void index(int , int , int , char* , char* );

void search(int , int , int , int , float , int , int,  char* , char* , char* , char* );

void ground_truth(int , int , int , char* , char* , char* );


int main(int argc, char **argv) {
	
	int alg = atoi(argv[1]);
	int n_pts = atoi(argv[2]);
	int n_query = atoi(argv[3]);
	int n_sample = atoi(argv[4]);
	int d = atoi(argv[5]);
	int topk = atoi(argv[6]);
	
	//------------------------------
	float delta= atof(argv[7]);
	int l0 = atoi(argv[8]);
	int flag = atoi(argv[9]);
	//---------------------------
	
    char* data_path = argv[10];
	char* query_path = argv[11];
	char* gt_path = argv[12];
	char* index_path=  argv[13];
    
	if(alg == 0){
		ground_truth(n_pts, d, n_query, data_path, query_path, gt_path);
	}
	else if(alg == 1){
		index(n_pts, d, n_sample, data_path, index_path);
	}
	else{
		search(n_pts, n_query, d, topk, delta, l0, flag, query_path, data_path, gt_path, index_path);
	}

};
