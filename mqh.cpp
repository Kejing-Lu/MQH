#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <vector>
#include "mqh_lib/visited_list_pool.h"
#include <cmath>
#include <cassert>
#include <unordered_set>
#include <algorithm>
#include <limits.h>
#define L 256

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif
#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

using namespace std;
using namespace hnswlib;
class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

static inline int fast_count(unsigned long a, unsigned long b){   
	unsigned long u = a ^ b;
    int count = _popcnt64(u);
    return count;	
}   

   float compare_short(const float* a, const float* b, unsigned size) {
      float dot0, dot1, dot2, dot3;
      const float* last = a + size;
      const float* unroll_group = last - 3;
      float result = 0;
      while (a < unroll_group) {
          dot0 = a[0] * b[0];
          dot1 = a[1] * b[1];
          dot2 = a[2] * b[2];
          dot3 = a[3] * b[3];
          result += dot0 + dot1 + dot2 + dot3;
          a += 4;
          b += 4;
      }
      while (a < last) {
          result += *a++ * *b++;
      }
      return result;  
    }

   float compare_ip(const float* a, const float* b, unsigned size) {
      float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
      #define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
          tmp1 = _mm256_loadu_ps(addr1);\
          tmp2 = _mm256_loadu_ps(addr2);\
          tmp1 = _mm256_mul_ps(tmp1, tmp2); \
          dest = _mm256_add_ps(dest, tmp1);

	  __m256 sum;
   	  __m256 l0, l1;
   	  __m256 r0, r1;
      unsigned D = (size + 7) & ~7U;
      unsigned DR = D % 16;
      unsigned DD = D - DR;
   	  const float *l = a;
   	  const float *r = b;
      const float *e_l = l + DD;
   	  const float *e_r = r + DD;
      float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};

      sum = _mm256_loadu_ps(unpack);
      if(DR){AVX_DOT(e_l, e_r, sum, l0, r0);}

      for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
	    AVX_DOT(l, r, sum, l0, r0);
	    AVX_DOT(l + 8, r + 8, sum, l1, r1);
      }
      _mm256_storeu_ps(unpack, sum);
      result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
#else
#ifdef __SSE2__
      #define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
          tmp1 = _mm128_loadu_ps(addr1);\
          tmp2 = _mm128_loadu_ps(addr2);\
          tmp1 = _mm128_mul_ps(tmp1, tmp2); \
          dest = _mm128_add_ps(dest, tmp1);
      __m128 sum;
      __m128 l0, l1, l2, l3;
      __m128 r0, r1, r2, r3;
      unsigned D = (size + 3) & ~3U;
      unsigned DR = D % 16;
      unsigned DD = D - DR;
      const float *l = a;
      const float *r = b;
      const float *e_l = l + DD;
      const float *e_r = r + DD;
      float unpack[4] __attribute__ ((aligned (16))) = {0, 0, 0, 0};

      sum = _mm_load_ps(unpack);
      switch (DR) {
          case 12:
          SSE_DOT(e_l+8, e_r+8, sum, l2, r2);
          case 8:
          SSE_DOT(e_l+4, e_r+4, sum, l1, r1);
          case 4:
          SSE_DOT(e_l, e_r, sum, l0, r0);
        default:
          break;
      }
      for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
          SSE_DOT(l, r, sum, l0, r0);
          SSE_DOT(l + 4, r + 4, sum, l1, r1);
          SSE_DOT(l + 8, r + 8, sum, l2, r2);
          SSE_DOT(l + 12, r + 12, sum, l3, r3);
      }
      _mm_storeu_ps(unpack, sum);
      result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else
      float dot0, dot1, dot2, dot3;
      const float* last = a + size;
      const float* unroll_group = last - 3;

      while (a < unroll_group) {
          dot0 = a[0] * b[0];
          dot1 = a[1] * b[1];
          dot2 = a[2] * b[2];
          dot3 = a[3] * b[3];
          result += dot0 + dot1 + dot2 + dot3;
          a += 4;
          b += 4;
      }
      while (a < last) {
          result += *a++ * *b++;
      }
#endif
#endif
#endif
      return result;
    }

int comp_float(const void*a,const void*b)
{
return *(float*)a-*(float*)b;
}

int comp_int(const void*a,const void*b)
{
return *(int*)a-*(int*)b;
}

struct elem{
	int id;
    float val;
};

struct Q_elem{
	unsigned char id1;
    unsigned char id2;
	int num;
};

struct elem2{
	int id;
    float val;
	float sort_val;
};

struct Neighbor{
	int id;
	float distance;
};
		
struct Neighbor2{
	int id;
	float val;
	float sort_val;
};		
		
int Elemcomp_a(const void*a, const void*b)
{  
  elem x1 = *((elem*) b);
  elem x2 = *((elem*) a);

  if(x1.val > x2.val)
       return -1;
  else {return 1;}
}

int Elemcomp_d(const void*a, const void*b)
{  
  elem x1 = *((elem*) b);
  elem x2 = *((elem*) a);

  if(x1.val > x2.val)
       return 1;
  else {return -1;}
}

int Elemcomp2(const void*a, const void*b)
{
	
  elem2 x1 = *((elem2*) b);
  elem2 x2 = *((elem2*) a);

  if(x1.sort_val > x2.sort_val)
       return -1;
  else {return 1;}
}

float uniform(						
	float min,							
	float max)							
{
	int   num  = rand();
	float base = (float) RAND_MAX - 1.0F;
	float frac = ((float) num) / base;

	return (max - min) * frac + min;
}

float gaussian(						
	float mean,							
	float sigma)						
{
	float v1 = -1.0f;
    float v2 = -1.0f;
	float s  = -1.0f;
	float x  = -1.0f;

	do {
		v1 = 2.0F * uniform(0.0F, 1.0F) - 1.0F;
		v2 = 2.0F * uniform(0.0F, 1.0F) - 1.0F;
		s = v1 * v1 + v2 * v2;
	} while (s >= 1.0F);
	x = v1 * sqrt (-2.0F * log (s) / s);

	x = x * sigma + mean; 			
	return x;
}

static inline int findinPool(elem *addr, int K, float val) {
    int left=0,right=K-1;
    if(addr[left].val>val){
        return left;
    }

    while(left<right-1){
        int mid=(left+right)/2;
        if(addr[mid].val>val)right=mid;
        else left=mid;
    }
    return right;
}		
		
static inline int InsertIntoPool (Neighbor *addr, unsigned K, Neighbor nn) {
    int left=0,right=K-1;
    if(addr[left].distance>nn.distance){
        memmove((char *)&addr[left+1], &addr[left],K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if(addr[right].distance<nn.distance){
        addr[K] = nn;
        return K;
    }
    while(left<right-1){
        int mid=(left+right)/2;
        if(addr[mid].distance>nn.distance)right=mid;
        else left=mid;
    }

    while (left > 0){
        if (addr[left].distance < nn.distance) break;
        if (addr[left].id == nn.id) return K + 1;
        left--;
    }
    if(addr[left].id == nn.id||addr[right].id==nn.id)return K+1;
    memmove((char *)&addr[right+1], &addr[right],(K-right) * sizeof(Neighbor));
    addr[right]=nn;
    return right;
} 
		
static inline int InsertIntoPool2 (Neighbor2 *addr, unsigned K, Neighbor2 nn) {
    int left=0,right=K-1;
    if(addr[left].sort_val>nn.sort_val){
        memmove((char *)&addr[left+1], &addr[left],K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
	
    if(addr[right].sort_val<nn.sort_val){
        addr[K] = nn;
        return K;
    }
    while(left<right-1){
        int mid=(left+right)/2;
        if(addr[mid].sort_val>nn.sort_val)right=mid;
        else left=mid;
    }

    while (left > 0){
        if (addr[left].sort_val < nn.sort_val) break;
        if (addr[left].id == nn.id) return K + 1;
        left--;
    }
	
    if(addr[left].id == nn.id||addr[right].id==nn.id)return K+1;
    memmove((char *)&addr[right+1], &addr[right],(K-right) * sizeof(Neighbor));
    addr[right]=nn;
    return right;
} 		

void K_means(float** train, double** vec, int n_sample, int d){
	
	int seed_ = 1;
    int cur_obj = 0;
    int* array_ = new int[L];
    bool flag_ = false;
	
	
    for(int i = 0; i < L; i++){                             
        srand(seed_);
        seed_++;
        int l = rand() % n_sample;
        for(int j = 0; j < d; j++){
		    vec[i][j] = train[l][j];
		}
        flag_ = false;
            
		for(int j = 0; j < cur_obj; j++){ 
            if(l == array_[j]){
                i--; 
				flag_ = true; 
				break;
            } 
        }
        if(flag_ == false){
            array_[cur_obj] = l; 
			cur_obj++; 	
        }
	}
	
	delete[] array_;
	
	float sum, min_sum;
	int vec_id;
	int* pvec = new int[n_sample];
	int* count = new int[L];
	int ROUND = 20;
	
	for(int k = 0; k < ROUND; k++){
        for(int j = 0; j < L; j++)
		    count[j] = 0;
	
		for(int j = 0; j < n_sample; j++){  			
			for(int l = 0; l < L; l++){
				sum = 0;
                for(int i = 0; i < d; i++){
			        sum += (train[j][i] - vec[l][i]) * (train[j][i] - vec[l][i]);
                }
                if( l == 0) {min_sum = sum; vec_id = 0;}
                else if(sum < min_sum) {min_sum = sum; vec_id = l;}				
		    }
            pvec[j] = vec_id;
	        count[pvec[j]]++;			
		}
                    	
		for(int j = 0; j < n_sample; j++){
			for(int i = 0; i < d; i++){
			    vec[ pvec[j] ][i] = 0; 
			}	
		}
                    
		for(int j = 0; j < n_sample; j++){  
			for(int i = 0; i < d; i++){
			    vec[ pvec[j] ][i] += train[j][i]; 
			}	
		}
		
		for(int j = 0; j < L; j++){
			if(count[j] == 0) continue;
			for(int i = 0; i < d; i++){
                vec[j][i] = vec[j][i] / count[j];			
			}
		}	
    }
    delete[] count;
    delete[] pvec;	
}	

float calc_norm(float* array, int d){   
	float sum = 0;
    for(int i = 0; i < d; i++){
	    sum += array[i] * array[i];
    }
	return sqrt(sum);
}		

void select_sample(float** data, float** train, int n_pts, int size, int dim){
	int interval = n_pts / size;
	
	int cur_obj = 0;
	for(int i = 0; i < size; i++){
		for(int j = 0; j < dim; j++){
		    train[i][j] = data[cur_obj][j];
		}
        cur_obj += interval;		
	}
}

static inline float pq_dist(unsigned char* a, float** b, int size){
	float sum = 0;
	for(int i = 0; i < size; i++){
		unsigned char x = a[i];
		sum += b[i][x];
	}
	return sum;
}	
		
	void ground_truth(int n_pts, int d, int n_query, char* path_data, char* query_data, char* path_gt){
		int maxk = 100;
		float** data = new float*[n_pts];
		for(int i = 0; i < n_pts; i++)
			data[i] = new float[d];
		
		ifstream input(path_data, ios::binary);
		for (int i = 0; i < n_pts; i++) {
            int t;
            input.read((char *) &t, 4); 
            input.read((char *) (data[i]), sizeof(float) * d);
        }
		
		input.close();
		float** query = new float*[n_query];
		float* u = new float[n_query];
		
		for(int i = 0; i < n_query; i++)
			query[i] = new float[d];
		
		ifstream input_query(query_data, ios::binary);
		for (int i = 0; i < n_query; i++) {
            int t;
            input_query.read((char *) &t, 4);
            input_query.read((char *) (query[i]), 4 * d);
			input_query.read((char *) &(u[i]), 4);
        }
		
		input_query.close();

        ofstream outputGT(path_gt, ios::binary);	
		
		elem* results = new elem[n_pts];
		for (int i = 0; i < n_query; i++) {
			for(int j = 0; j < n_pts; j++){
		        float distance = compare_ip( data[j], query[i], d) - u[i];
				if(distance < 0) distance = -1 * distance;
				
				results[j].id = j;
				results[j].val = distance;
			}
			qsort(results, n_pts, sizeof(elem), Elemcomp_a);   
			
			outputGT.write((char *) &i, sizeof(int));
			for(int j = 0; j < maxk; j++){
                outputGT.write((char *) (results[j].id), sizeof(int));
			}			
		}
		outputGT.close();
	}
	
	void index(int n_pts, int d, int n_sample, char* path_data, char* index_data){
		int M2 = 16;
		int level = 4;
	    int size = M2 * level;
	
		int m_level = 1;
		int m_num = 64;
		int m = m_level * m_num;
		
		int d_org = d;
		int d_supp;
		if( d % M2 == 0)
			d_supp = 0;
		else {d_supp = M2 - d % M2;}
		d = d + d_supp;
		int d2 = d / M2;
		
		float** data = new float*[n_pts];
		for(int i = 0; i < n_pts; i++)
			data[i] = new float[d];
		
		for(int i = 0; i < n_pts; i++)
		    for(int j = 0; j < d; j++)
				data[i][j] = 0;		
		
		ifstream input(path_data, ios::binary);
		for (int i = 0; i < n_pts; i++) {
            float t;
            input.read((char *) (data[i]), sizeof(float) * d_org);
			input.read((char *) &t, sizeof(float));  // read additional dimension
		    if((int) t != 1) cout<<"dimensin error"<<endl;
        }
		
		ofstream output(index_data, ios::binary);
		
		input.close();
		
		StopW stopw = StopW();
		
		float sum = 0;
		float min_sum; int min_id;
		float* norm = new float[n_pts];

		float** residual_vec = new float* [n_pts];
		for(int i = 0; i < n_pts; i++){	
		    residual_vec[i] = new float[d];
		}
		
		double** vec_1 = new double* [L];
		for(int i = 0; i < L; i++)
			vec_1[i] = new double[d/2];
		
		double** vec_2 = new double* [L];
		for(int i = 0; i < L; i++)
			vec_2[i] = new double[d/2];		
		
		float** train = new float*[n_sample];
		for(int i = 0; i < n_sample; i++)
			train[i] = new float[d];
		
		for(int i = 0; i < n_pts; i++){
            norm[i] = calc_norm(data[i], d);
		}

		select_sample(data, train, n_pts, n_sample, d);
		
		float** train1 = new float* [n_sample];
		for(int i = 0; i < n_sample; i++)
			train1[i] = new float[d/2];	

		float** train2 = new float* [n_sample];
		for(int i = 0; i < n_sample; i++)
			train2[i] = new float[d/2];			
		
	    for(int i = 0; i < n_sample; i++){
			for(int j = 0; j < d/2; j++){
				train1[i][j] = train[i][j];
			}
			
			for(int j = 0; j < d/2; j++){
			    train2[i][j] = train[i][j + d/2];
			}
		}
		
		K_means(train1, vec_1, n_sample, d/2);

		K_means(train2, vec_2, n_sample, d/2);
	
		int* count = new int[L*L];
        unsigned char* vec_id1 = new unsigned char[n_pts];
		unsigned char* vec_id2 = new unsigned char[n_pts];		
		
		for(int i = 0; i < L*L; i++)
			count[i] = 0;
		
		for(int i = 0; i < n_pts; i++){ 		
            for(int j = 0; j < L; j++){
				sum = 0;
				for(int l = 0; l < d/2; l++){
					sum += (data[i][l] - vec_1[j][l]) * (data[i][l] - vec_1[j][l]);
				}
				
				if(j == 0) {min_sum = sum; min_id = 0;}
                else {
					if(sum < min_sum){
					    min_sum = sum; 
                        min_id = j;						
					}	
				}   				
			}
			vec_id1[i] = min_id;

            for(int j = 0; j < L; j++){
				sum = 0;
				for(int l = 0; l < d/2; l++){
				    sum += (data[i][l + d/2] - vec_2[j][l]) * (data[i][l + d/2] - vec_2[j][l]);
				}
				
				if(j == 0) {min_sum = sum; min_id = 0;}
                else {
					if(sum < min_sum){
					    min_sum = sum; 
                        min_id = j;						
					}	
				}   				
			}
			vec_id2[i] = min_id;						
			count[ vec_id1[i] * L + vec_id2[i] ]++;
		}

        for(int i = 0; i < L; i++){
			for(int j = 0; j < d/2; j++){
				float x = vec_1[i][j];
		        output.write( (char *) &x, sizeof(float));
			}		
		}			
					
        for(int i = 0; i < L; i++){
			for(int j = 0; j < d/2; j++){
				float x = vec_2[i][j];
		        output.write( (char *) &x, sizeof(float));
			}		
		}

		int n_cand1 = 0;		
		int* map_table = new int[L * L];
		for(int i = 0; i < L*L; i++){
			map_table[i] = -1;			
			if(count[i] > 0) n_cand1++;
		}
		
		elem** array1 = new elem* [n_cand1];
		int* n_temp = new int[n_cand1];
		for(int i = 0; i < n_cand1; i++) n_temp[i] = 0;
		Q_elem* pq_M2 = new Q_elem[n_cand1];
		
		n_cand1 = 0; 
		for(int i = 0; i < L*L; i++){ 
			if(count[i] > 0){
				array1[n_cand1] = new elem[count[i]];
				map_table[i] = n_cand1;
				
				pq_M2[n_cand1].id1 = i / L;
				pq_M2[n_cand1].id2 = i % L;
				pq_M2[n_cand1].num = count[i];
				
				n_cand1++;
			}	
		}
		
		output.write( (char *) &(n_cand1), 4);
        for(int i = 0; i < n_cand1; i++){		
		    output.write( (char *) &(pq_M2[i].id1), sizeof(unsigned char));
			output.write( (char *) &(pq_M2[i].id2), sizeof(unsigned char));
            output.write( (char *) &(pq_M2[i].num), sizeof(int));				
		}				
	
		int s = 0;
		int* count1 = new int[n_cand1]; 
		for(int i = 0; i < L*L; i++){
			if(count[i] > 0){
                count1[s] = count[i];
				s++;
			}	
		}		
		
		float* vec = new float[d];
		
		float ttest2;
        for(int i = 0; i < n_pts; i++){		
			int temp =  vec_id1[i] * L + vec_id2[i];
			int table_id = map_table[temp];
			array1[table_id][ n_temp[table_id] ].id = i;
			
	        for(int j = 0; j < d/2; j++){
		        vec[j] = vec_1[ vec_id1[i] ][j];
	        }
			
	        for(int j = 0; j < d/2; j++){
		        vec[j + d/2] = vec_2[ vec_id2[i] ][j];
	        }			
				
			for(int j = 0; j < d; j++){
				residual_vec[i][j] = data[i][j] - vec[j];
			}
						
			array1[table_id][ n_temp[table_id] ].val = 0;
			for(int j = 0; j < d; j++)
			    array1[table_id][ n_temp[table_id] ].val += residual_vec[i][j] * residual_vec[i][j]; 
				
            array1[table_id][ n_temp[table_id] ].val = sqrt( array1[table_id][ n_temp[table_id] ].val );				
			n_temp[table_id]++;	
		}
		
	    for(int i = 0; i < n_cand1; i++){
		    qsort(array1[i], count1[i], sizeof(elem), Elemcomp_d);    
		}
		
        for(int i = 0; i < n_cand1; i++){		
            for(int j = 0; j < count1[i]; j++){
				output.write( (char *) &(array1[i][j].id), sizeof(int));
                output.write( (char *) &(array1[i][j].val), sizeof(float));				
			}			
		}								
		
		float* norm2 = new float[n_pts];
		for(int i = 0; i < n_pts; i++){
			norm2[i] = calc_norm(residual_vec[i], d);
		}
		
		bool* zero_flag = new bool[n_pts];
		for(int i = 0; i < n_pts; i++){
		    zero_flag[i] = false;
		}
		
		float min_float = 0.0000001;
		
	    for(int i = 0; i < n_pts; i++){

			if(norm2[i] < min_float){			
				zero_flag[i] == true;
				
				residual_vec[i][0] = 1;
				for(int j = 1; j < d; j++){
				    residual_vec[i][j] = 0;
			    }				
			}
			else{
				for(int j = 0; j < d; j++){
				    residual_vec[i][j] = residual_vec[i][j] / norm2[i];
			    }
			}	
	    }
			
		double*** vec2 = new double**[size];
		for(int i = 0; i < size; i++)
			vec2[i] = new double*[L];		
		for(int i = 0; i < size; i++){
			for(int j = 0; j < L; j++){
				vec2[i][j] = new double[d2];
			}
		}

        float** residual_pq = new float*[n_sample];
        for(int i = 0; i < n_sample; i++){
			residual_pq[i] = new float[d2];
		}		

        unsigned char** pq_id = new unsigned char*[n_pts];
        for(int i = 0; i < n_pts; i++){
			pq_id[i] = new unsigned char[M2];
		}	

		float** proj_array = new float* [m];
		for (int i = 0; i < m; ++i) {
		    proj_array[i] = new float[d];
		    for (int j = 0; j < d; ++j) {
			    proj_array[i][j] = gaussian(0.0f, 1.0f);  
		    }
	    }
		
		for (int i = 0; i < m; ++i) 
		     output.write( (char *) (proj_array[i]), sizeof(float) * d);	

		for(int i = 0; i < n_pts; i++){
		    for(int j = 0; j < m_level; j++){
				unsigned long code_num = 0;
				for(int l = 0; l < m_num; l++){
					float ssum = 0;
					for(int ll = 0; ll < d; ll++)
						ssum +=  residual_vec[i][ll] * proj_array[j * m_num + l][ll];
						
					if(ssum >= 0){
						code_num += 1;
					}
						
					if(l < m_num - 1)
						code_num = code_num<<1;
						
				}
				output.write( (char *) &code_num, sizeof(unsigned long));	
			}
		}

     	float* proj_temp = new float[m];
        float** proj_val = new float* [n_pts];
		for(int i = 0; i < n_pts; i++)
			proj_val[i] = new float[m];
 
        float* test_vec = new float[d]; 
        for(int k = 0; k < level; k++){	
		    for(int i = 0; i < M2; i++){
				int ccount = 0;
                for(int j = 0; j < n_pts; j++){
					if(zero_flag[j] == true){
						continue;
					}	
				    for(int l = 0; l < d2; l++){
					     residual_pq[ccount][l] = residual_vec[j][ i * d2 + l ];
				    }
					ccount++;
					if(ccount >=n_sample) break;	
			    }			
                K_means(residual_pq, vec2[k * M2 + i], n_sample, d2);    
		    }
					
		    for(int n = 0; n < n_pts; n++){
		        for(int i = 0; i < M2; i++){   			
		            for(int j = 0; j < L; j++){
				        sum = 0;
			            for(int l = 0; l < d2; l++){
				            sum += (residual_vec[n][i * d2 + l] - vec2[k * M2 + i][j][l]) * (residual_vec[n][i * d2 + l] - vec2[k * M2 + i][j][l]);
				        }
		    
			            if(j == 0) {min_sum = sum; min_id = 0;}
                        else{
				            if(sum < min_sum){
					            min_sum = sum; 
                                min_id = j;						
				            } 	
			            }
			        }
                    pq_id[n][i] = min_id;			
		        }
		    }	

		    for(int n = 0; n < n_pts; n++){
			    for(int j = 0; j < M2; j++){
				    int temp_M = k * M2 + j;
				    for(int l = 0; l < d2; l++){
					    residual_vec[n][ j * d2 + l] = residual_vec[n][ j * d2 + l] - vec2[temp_M][ pq_id[n][j] ][l];
						test_vec[ j * d2 + l] = vec2[temp_M][ pq_id[n][j] ][l];
				    }
			    }
				float sum = 0;
				for(int j = 0; j < d; j++){
					sum += residual_vec[n][j] * residual_vec[n][j];
				}
				norm2[n] = sqrt(sum);		
				if(norm2[n] < min_float) zero_flag[n] = true;
		    }

            for(int i = 0; i < n_pts; i++){            	
			    output.write( (char *) &(norm2[i]), sizeof(float));
				output.write( (char *) (pq_id[i]), M2);					
		    }	

		    for(int i = 0; i < n_pts; i++){
				for(int j = 0; j < m_level; j++){
					unsigned long code_num = 0;
					for(int l = 0; l < m_num; l++){
						float ssum = 0;
						for(int ll = 0; ll < d; ll++)
							ssum +=  residual_vec[i][ll] * proj_array[j * m_num + l][ll];
						
						if(ssum >= 0){
							code_num += 1;
						}
						
						if(l < m_num - 1)
						    code_num = code_num<<1;
						
					}
					output.write( (char *) &code_num, sizeof(unsigned long));	
				}			
		    }
		}
		
		for(int i = 0; i < size; i++){
			for(int j = 0; j < L; j++){
				for(int l = 0; l < d2; l++){
				    float x = vec2[i][j][l];
                    output.write( (char *) &x, sizeof(float));
				}
			}				
		}
		
        output.close();	

		float time_us_indexing = stopw.getElapsedTimeMicro();
        cout << time_us_indexing / 1000 / 1000 << " s" << "\n";
	}

    float Quantile(float* table, float a, int size){
	    int i = 0;
		for(i = 0; i < size; i++){
			if(a < table[i])
				break;
		}
		return (1.0f * i / 100);
	}

    void search(int n_pts, int n_query, int d, int topk, float delta, int l0, int flag_, char* query_data, char* base_data, char* path_gt, char* path_index){
		int real_topk = topk;
		topk = 100;
				
		int maxk = 100;
		int thres_pq =  n_pts/ 10;   
		int thres_pq2 =  n_pts / 2;
		
		int n_exact = 2000;
		
	    std::vector<Neighbor> retset(topk + 1);
		std::vector<Neighbor> retset2(n_exact + 1);
		int M2 = 16;
		int level = 4;
	    int size = M2 * level;
        int m_level = 1;
		int m_num = 64;
		
		int m = m_level * m_num;

        float delta1;
		if(flag_ == 1)
			delta1 = 1;
		else{
			delta1 = delta;
		}
		
		int offset0 = l0;
		
		int m2 = m;
		float half_m2 = m2 /2; 
		
		VisitedListPool *visited_list_pool_ = new VisitedListPool(1, n_pts);
		
		int d_org = d;
		int d_supp;
		if( d % M2 == 0)
			d_supp = 0;
		else {d_supp = M2 - d % M2;}
		d = d + d_supp;
		int d2 = d / M2;
				
		float PI = 3.1415926535;
		
		int cosine_inv = 100;
		int* cosine_table = new int[cosine_inv];
		
		for(int i = 0; i < cosine_inv; i++){
			cosine_table[i] = m * acos(  1.0f * i / cosine_inv) / PI + offset0; 
			if(cosine_table[i] > m) cosine_table[i] = m;
		}
		
		float** proj_array = new float* [m];
		for (int i = 0; i < m; ++i) 
		    proj_array[i] = new float[d];
		
		float epsilon = 0.99999;
		float alpha = 0.673;
			
		float temp = sqrt(log(1 / epsilon) / 2 / m);  
		int table_size = 170;
		float* quantile_table = new float[table_size];
		quantile_table[0] = 0.5;
		quantile_table[1] = 0.504;
		quantile_table[2] = 0.508;
		quantile_table[3] = 0.512;
		quantile_table[4] = 0.516;
		quantile_table[5] = 0.52;
		quantile_table[6] = 0.524;
		quantile_table[7] = 0.528;
		quantile_table[8] = 0.532;
		quantile_table[9] = 0.536;
		
		quantile_table[10] = 0.54;
		quantile_table[11] = 0.544;
		quantile_table[12] = 0.548;
		quantile_table[13] = 0.552;
		quantile_table[14] = 0.556;
		quantile_table[15] = 0.56;
		quantile_table[16] = 0.564;
		quantile_table[17] = 0.568;
		quantile_table[18] = 0.571;
		quantile_table[19] = 0.575;	
		
		quantile_table[20] = 0.58;
		quantile_table[21] = 0.583;
		quantile_table[22] = 0.587;
		quantile_table[23] = 0.591;
		quantile_table[24] = 0.595;
		quantile_table[25] = 0.599;
		quantile_table[26] = 0.603;
		quantile_table[27] = 0.606;
		quantile_table[28] = 0.61;
		quantile_table[29] = 0.614;		
		
		quantile_table[30] = 0.618;
		quantile_table[31] = 0.622;
		quantile_table[32] = 0.626;
		quantile_table[33] = 0.63;
		quantile_table[34] = 0.633;
		quantile_table[35] = 0.637;
		quantile_table[36] = 0.641;
		quantile_table[37] = 0.644;
		quantile_table[38] = 0.648;
		quantile_table[39] = 0.652;	

		quantile_table[40] = 0.655;
		quantile_table[41] = 0.659;
		quantile_table[42] = 0.663;
		quantile_table[43] = 0.666;
		quantile_table[44] = 0.67;
		quantile_table[45] = 0.674;
		quantile_table[46] = 0.677;
		quantile_table[47] = 0.681;
		quantile_table[48] = 0.684;
		quantile_table[49] = 0.688;	
		
		quantile_table[50] = 0.692;
		quantile_table[51] = 0.695;
		quantile_table[52] = 0.699;
		quantile_table[53] = 0.702;
		quantile_table[54] = 0.705;
		quantile_table[55] = 0.709;
		quantile_table[56] = 0.712;
		quantile_table[57] = 0.716;
		quantile_table[58] = 0.719;
		quantile_table[59] = 0.722;			
		
		quantile_table[60] = 0.726;
		quantile_table[61] = 0.729;
		quantile_table[62] = 0.732;
		quantile_table[63] = 0.736;
		quantile_table[64] = 0.74;
		quantile_table[65] = 0.742;
		quantile_table[66] = 0.745;
		quantile_table[67] = 0.749;
		quantile_table[68] = 0.752;
		quantile_table[69] = 0.755;		

		quantile_table[70] = 0.758;
		quantile_table[71] = 0.761;
		quantile_table[72] = 0.764;
		quantile_table[73] = 0.767;
		quantile_table[74] = 0.77;
		quantile_table[75] = 0.773;
		quantile_table[76] = 0.776;
		quantile_table[77] = 0.779;
		quantile_table[78] = 0.782;
		quantile_table[79] = 0.785;	

		quantile_table[80] = 0.788;
		quantile_table[81] = 0.791;
		quantile_table[82] = 0.794;
		quantile_table[83] = 0.797;
		quantile_table[84] = 0.8;
		quantile_table[85] = 0.802;
		quantile_table[86] = 0.805;
		quantile_table[87] = 0.808;
		quantile_table[88] = 0.811;
		quantile_table[89] = 0.813;

		quantile_table[90] = 0.816;
		quantile_table[91] = 0.819;
		quantile_table[92] = 0.821;
		quantile_table[93] = 0.824;
		quantile_table[94] = 0.826;
		quantile_table[95] = 0.829;
		quantile_table[96] = 0.832;
		quantile_table[97] = 0.834;
		quantile_table[98] = 0.837;
		quantile_table[99] =	0.839;
		
		quantile_table[100] = 0.841;
		quantile_table[101] = 0.844;
		quantile_table[102] = 0.846;
		quantile_table[103] = 0.849;
		quantile_table[104] = 0.851;
		quantile_table[105] = 0.853;
		quantile_table[106] = 0.855;
		quantile_table[107] = 0.858;
		quantile_table[108] = 0.85;
		quantile_table[109] = 0.862;

		quantile_table[110] = 0.864;
		quantile_table[111] = 0.867;
		quantile_table[112] = 0.869;
		quantile_table[113] = 0.871;
		quantile_table[114] = 0.873;
		quantile_table[115] = 0.875;
		quantile_table[116] = 0.877;
		quantile_table[117] = 0.879;
		quantile_table[118] = 0.881;
		quantile_table[119] = 0.883;

		quantile_table[120] = 0.885;
		quantile_table[121] = 0.887;
		quantile_table[122] = 0.889;
		quantile_table[123] = 0.891;
		quantile_table[124] = 0.893;
		quantile_table[125] = 0.894;
		quantile_table[126] = 0.896;
		quantile_table[127] = 0.898;
		quantile_table[128] = 0.9;
		quantile_table[129] =	0.902;		
	
		quantile_table[130] = 0.903;
		quantile_table[131] = 0.905;
		quantile_table[132] = 0.907;
		quantile_table[133] = 0.908;
		quantile_table[134] = 0.910;
		quantile_table[135] = 0.912;
		quantile_table[136] = 0.913;
		quantile_table[137] = 0.915;
		quantile_table[138] = 0.916;
		quantile_table[139] =	0.918;

		quantile_table[140] = 0.919;
		quantile_table[141] = 0.921;
		quantile_table[142] = 0.922;
		quantile_table[143] = 0.924;
		quantile_table[144] = 0.925;
		quantile_table[145] = 0.927;
		quantile_table[146] = 0.928;
		quantile_table[147] = 0.929;
		quantile_table[148] = 0.931;
		quantile_table[149] =	0.932;

		quantile_table[150] = 0.933;
		quantile_table[151] = 0.935;
		quantile_table[152] = 0.936;
		quantile_table[153] = 0.937;
		quantile_table[154] = 0.938;
		quantile_table[155] = 0.939;
		quantile_table[156] = 0.941;
		quantile_table[157] = 0.942;
		quantile_table[158] = 0.943;
		quantile_table[159] =	0.944;

		quantile_table[160] = 0.945;
		quantile_table[161] = 0.946;
		quantile_table[162] = 0.947;
		quantile_table[163] = 0.948;
		quantile_table[164] = 0.95;
		quantile_table[165] = 0.951;
		quantile_table[166] = 0.952;
		quantile_table[167] = 0.953;
		quantile_table[168] = 0.954;
		quantile_table[169] =	0.955;		
		
		float coeff = Quantile(quantile_table, 0.5 * temp + 0.75, table_size);
	
		temp = sqrt(log(1 / epsilon) / 2 / m); 
	    float coeff2 = Quantile(quantile_table, 0.5 * temp + 0.75, table_size);
		
		float max_coeff = Quantile(quantile_table, 0.5 * (1-temp) + 0.5, table_size);
	
		int max_inv = 1000;
		unsigned char* tab_inv = new unsigned char[max_inv];
		
		float ratio = (max_coeff - coeff2) / max_inv;
		for(int i = 0; i < max_inv; i++){
			float temp2 = coeff2 + i * ratio;
					
			int temp3 = temp2 * 100;
			if(temp3 >= table_size) temp3 = table_size - 1;   
			temp2 = 2 * (quantile_table[temp3] - 0.5) + temp;			
			

			tab_inv[i] =  temp2 * m + 1;
			if (tab_inv[i] > m) tab_inv[i] = m; 
		}	
		
		float* ttab = new float[table_size];
		for(int i = 0; i < table_size; i++){
			ttab[i] = (quantile_table[i] - 0.5) * 2 * m; 
		}
		
		int *massQA = new int[n_query * maxk];
	
		FILE *fp = fopen(path_gt, "r");
        if (!fp) { printf("Could not open %s\n", path_gt); exit(0); }

        int tmp1 = -1, tmp2 = -1;
        fscanf(fp, "%d,%d\n", &tmp1, &tmp2); assert(tmp1==n_query && tmp2==maxk);

        for (int i = 0; i < n_query; ++i) {
            fscanf(fp, "%d", &tmp1);
            for (int j = 0; j < maxk; ++j) {
			    float tmp3;
                fscanf(fp, ",%d,%f", massQA + maxk * i + j, &tmp3);
            }
            fscanf(fp, "\n");
        }
        fclose(fp);
		
		int **search_result = new int*[n_query];
		for(int i = 0; i < n_query; i++) search_result[i] = new int[topk];

		float** data = new float*[n_pts];
		for(int i = 0; i < n_pts; i++)
			data[i] = new float[d];
		
		for(int i = 0; i < n_pts; i++)
			for(int j = 0; j < d; j++)
				data[i][j] = 0;
		
		ifstream input_data(base_data, ios::binary);
		for (int i = 0; i < n_pts; i++) {
            float t;
            input_data.read((char *) (data[i]), 4 * d_org);
			input_data.read((char *) &t, sizeof(float));
        }
		
		input_data.close();

		float** query = new float*[n_query];
		float* u = new float[n_query];
		
		for(int i = 0; i < n_query; i++)
			query[i] = new float[d];
		
		for(int i = 0; i < n_query; i++)
			for(int j = 0; j < d; j++)
				query[i][j] = 0;		
		
		ifstream input_query(query_data, ios::binary);
		for (int i = 0; i < n_query; i++) {

            input_query.read((char *) (query[i]), sizeof(float) * d_org);
			
			float norm_query = calc_norm(query[i], d);
			
			for(int j = 0; j < d; j++) query[i][j] = query[i][j] / norm_query;
			
			input_query.read((char *) &(u[i]), sizeof(float));
			
			u[i] = -1 * u[i] / norm_query;
        }
		
		input_query.close();
			
        ifstream input_index(path_index, ios::binary);  
		
		float** vec1 = new float* [L];
		for(int i = 0; i < L; i++)
			vec1[i] = new float[d/2];
		
		float** vec2 = new float* [L];
		for(int i = 0; i < L; i++)
			vec2[i] = new float[d/2];	

	    for(int i = 0; i < L; i++){	
		    for(int j = 0; j < d / 2; j++){
                float t;				
			    input_index.read( (char *) &t, sizeof(float));
		        vec1[i][j] = t;
			}
	    }			
					
	    for(int i = 0; i < L; i++){		
		    for(int j = 0; j < d / 2; j++){
                float t;				
			    input_index.read( (char *) &t, sizeof(float));
		        vec2[i][j] = t;
			}
	    }	

        int n_cand1;
		input_index.read( (char *) &(n_cand1), 4);
		
		Q_elem* pq_M2 = new Q_elem[n_cand1];
	
		int* count = new int[n_cand1];
        for(int i = 0; i < n_cand1; i++){		
		    input_index.read( (char *) &(pq_M2[i].id1), 1);   
			input_index.read( (char *) &(pq_M2[i].id2), 1);
			
		    input_index.read( (char *) &(count[i]), sizeof(int));
            pq_M2[i].num = count[i];		
		}

        int** coarse_index = new int*[n_cand1];
        for(int i = 0; i < n_cand1; i++) coarse_index[i] = new int[count[i]];

		char** index_ = (char **) malloc(sizeof(void *) * n_cand1);

	    int size_per_element_ = sizeof(int) + 2 * sizeof(float) + sizeof(unsigned long) * m_level + level * (M2 + sizeof(float) + sizeof(unsigned long) * m_level);
		int proj_per_element_ = level * m * sizeof(float);
		 
        for(int i = 0; i < n_cand1; i++){
			index_[i] =  (char *) malloc(count[i] * size_per_element_);
			
			char* cur_loc = index_[i];
		
			for(int j = 0; j < count[i]; j++){
				float b;
				input_index.read( (char *) &(coarse_index[i][j]), 4);  
                input_index.read( (char *) &(b), 4);
              
                memcpy(cur_loc, &(coarse_index[i][j]), sizeof(int));
			 	cur_loc += sizeof(int);
                memcpy(cur_loc, &b, sizeof(float));
                cur_loc += (size_per_element_ - sizeof(int));				
			}
		}

	    for(int i = 0; i < m; i++)
            input_index.read( (char *) (proj_array[i]), sizeof(float) * d);

        unsigned long** rough_code = new unsigned long*[n_pts];
        for(int i = 0; i < n_pts; i++) rough_code[i] = new unsigned long[m_level]; 

		for(int i = 0; i < n_pts; i++){
		    for(int j = 0; j < m_level; j++){
				input_index.read( (char *) &(rough_code[i][j]), sizeof(unsigned long));	
			}
		}

        for(int i = 0; i < n_cand1; i++){	
			char* cur_loc = index_[i];			
			for(int j = 0; j < count[i]; j++){
				cur_loc = index_[i] + j * size_per_element_ + sizeof(int) + 2 * sizeof(float);
				
				int b = coarse_index[i][j]; 
				
				for(int jj = 0; jj < m_level; jj++){
					memcpy(cur_loc, &(rough_code[b][jj]), sizeof(unsigned long));
				    cur_loc += sizeof(unsigned long);				
				} 				
			}
		}

        unsigned char** pq_id = new unsigned char*[n_pts];
        for(int i = 0; i < n_pts; i++){
			pq_id[i] = new unsigned char[M2];
		}	

        float* residual_norm = new float[n_pts];		
		
        unsigned long** code_num = new unsigned long*[n_pts];
        for(int i = 0; i < n_pts; i++) code_num[i] = new unsigned long[m_level]; 		
		
        for(int i = 0; i < level; i++){         
			for(int j = 0; j < n_pts; j++){ 
                input_index.read( (char *) &(residual_norm[j]), sizeof(float));				
				input_index.read( (char *) (pq_id[j]), M2);	
			}	
			
			for(int k =0; k < n_cand1; k++){
			    for(int j = 0; j < count[k]; j++){ 
			        int a = coarse_index[k][j];
				    char* cur_loc = index_[k] + j * size_per_element_ +  sizeof(int) + 2 * sizeof(float) + sizeof(unsigned long) * m_level + i * (M2 + sizeof(float) + sizeof(unsigned long) * m_level);
					
                    memcpy(cur_loc, &(residual_norm[a]), sizeof(float));
                    cur_loc += sizeof(float);					
				    for(int l = 0; l < M2; l++){
                        memcpy(cur_loc, &(pq_id[a][l]), 1);
						cur_loc += 1;
				    }    			
			    }
			}
			
			for(int j = 0; j < n_pts; j++){   
				for(int jj = 0; jj < m_level; jj++)
					input_index.read( (char *) &(code_num[j][jj]), sizeof(unsigned long));			
        	}

			for(int k =0; k < n_cand1; k++){
			    for(int j = 0; j < count[k]; j++){ 
			        int a = coarse_index[k][j];
				    char* cur_loc = index_[k] + j * size_per_element_ +  sizeof(int) + 2 * sizeof(float) + sizeof(unsigned long) * m_level + i * (M2 + sizeof(float) + sizeof(unsigned long) * m_level) + M2 + sizeof(float);
					
					for(int jj = 0; jj < m_level; jj++){
					     memcpy(cur_loc, &(code_num[a][jj]), sizeof(unsigned long));
						 cur_loc += sizeof(unsigned long);
					
					}   			
			    }
			}
			
		}
		
        float*** vec_pq = new float**[size];
        for(int i = 0; i < size; i++) vec_pq[i] = new float*[L];
 		for(int i = 0; i < size; i++)
			for(int j = 0; j < L; j++)
				vec_pq[i][j] = new float[d2];
		
		for(int i = 0; i < size; i++){
			for(int j = 0; j < L; j++){
                input_index.read( (char *) (vec_pq[i][j]), sizeof(float) * d2);
			}				
		}

        input_index.close();
		
		int half_d = d / 2;
		float half_m = m / 2;
		
		
		float* table_1 = new float[L];
		float* table_2 = new float[L];
		
		float*** table2 = new float**[level];
		for(int i = 0; i < level; i++){table2[i] = new float*[M2];}
		for(int i = 0; i < level; i++)
			for(int j = 0; j < M2; j++)
				table2[i][j] = new float[L];
		
        char* col_num = new char[n_pts]; 
		
		unsigned long* query_proj = new unsigned long[m_level];
			
		float INV1 = 1.5;
		int INV0 = 200;
	    float INV = INV1 / INV0;
			
		int nnum = 0;
		int nnum2 = 0;
		
	    int* tttest = new int[n_cand1];
        bool* is_zero = new bool[n_cand1];

        int left_inv0 = 100000000;
		int left_inv1 = 100000;
	
	    StopW stopw = StopW();

        int offset00 = sizeof(float) + m_level * sizeof(unsigned long);		
		int offset1 = m_level * sizeof(unsigned long);   
		int offset2 = M2 + m_level * sizeof(unsigned long);
		int offset3 = sizeof(float) + M2 + m_level * sizeof(unsigned long);
		
		int round1_offset = sizeof(float) + sizeof(unsigned long) * m_level + sizeof(float);
		
		for (int i = 0; i < n_query; i++) {
			
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;			
			
			float tau = u[i];
	        int num = 0;
			int num1 = 0;
			float* query1 = query[i];
                        
			int zero_count = 0;
			for(int j = 0; j < m_level; j++){
				query_proj[j] = 0;
				zero_count = 0;
				for(int jj = 0; jj < m_num; jj++){
                    float ttmp0 = compare_ip(query1, proj_array[j * m_num + jj], d);			
			
			        if(ttmp0 >= 0){
						query_proj[j] += 1;
						zero_count++;
					}
						
					if(jj < m_num - 1)
					    query_proj[j] = query_proj[j]<<1;
				}
				
			}
						
			float* query2 = query1 + half_d;
			
			for(int j = 0; j < L; j++){
			    float sum = compare_short(query1, vec1[j], half_d);			
                table_1[j] = sum;				
			}	
		
			for(int j = 0; j < L; j++){
			    float sum = compare_short(query2, vec2[j], half_d);
                table_2[j] = sum;				
			}

			for(int j = 0; j < level; j++){
				for(int l = 0; l < M2; l++){
					for(int k = 0; k < L; k++){
					    table2[j][l][k] = compare_short(query1 + l * d2, vec_pq[j * M2 + l][k], d2);  //!aware
					}
				}
			}

		    for(int j = 0; j < n_cand1; j++){
			    unsigned char a = pq_M2[j].id1;
				unsigned char b = pq_M2[j].id2;
				float tmp = table_1[a] + table_2[b] - tau;
				
				int tmp0;
			    if(tmp < 0){ 
					tmp0 = (int) (-1 * tmp * left_inv0);
					tmp0 = tmp0 - (tmp0 % left_inv1);
					tmp0 += j;
					is_zero[j] = true;
			    }
				
				else{
					tmp0 = (int) (tmp * left_inv0);
					tmp0 = tmp0 - (tmp0 % left_inv1);
					tmp0 += j;
					is_zero[j] = false;
				}
				tttest[j] = tmp0;
		    }	
			
            qsort(tttest, n_cand1, sizeof(int), comp_int);

			int s = 0;
			bool flag = false;
			
			int remain_size = size_per_element_ - sizeof(int) - ( 3 * sizeof(float) )- (m_level * sizeof(unsigned long));
			
		    for(int j = 0; j < n_cand1; j++){
				int a = tttest[j] % left_inv1;
              
				float cur_dist = 1.0f * (tttest[j] - a) / left_inv0;
				if(is_zero[a] == true) cur_dist = -1 * cur_dist;
				
				int b = count[a];
				
				char* cur_obj = index_[a];
		        for(int l = 0; l < b; l++){
					s++;
	                if (s > thres_pq) {flag = true; break;}
	            
					int x = *( (int *) cur_obj);   
					cur_obj += sizeof(int);
					float y = *( (float *) cur_obj );
					char* cur_obj_1 = cur_obj + sizeof(float);
					cur_obj = cur_obj_1 + round1_offset;
					
					float z = pq_dist( (unsigned char*) cur_obj, table2[0], M2);
					cur_obj += remain_size;
					
					float distance = cur_dist + z * y;
					
					memcpy(cur_obj_1, &distance, sizeof(float));
					
					if(distance < 0) distance = -1 * distance;
					
					Neighbor nn2;
					nn2.id = x; nn2.distance = distance;
					
					if(num1 == 0)
					    retset2[0] = nn2;
					else{
					    if(num1 >= n_exact){
							if(distance >= retset2[n_exact-1].distance) continue;
                            InsertIntoPool(retset2.data(), n_exact, nn2);
						}
				        else{
						    InsertIntoPool(retset2.data(), num1, nn2);
					    }
					}
					num1++;
					
				}
				if(flag == true) break;
		    }

			for(int j = 0; j < n_exact; j++){
				int ID = retset2[j].id;
				
				if (visited_array[ID] == visited_array_tag) continue;
				visited_array[ID] = visited_array_tag;
				
				float distance = compare_ip( data[ID], query1, d) - tau;
				if(distance < 0) distance = -1 * distance;
				Neighbor nn2;
				nn2.id = ID; nn2.distance = distance;
					
				if(num== 0)
					retset[0] = nn2;
				else{
				    if(num >= topk){
						if(distance >= retset[topk-1].distance) continue;
                        InsertIntoPool(retset.data(), topk, nn2);
					}
				    else{
						InsertIntoPool(retset.data(), num, nn2);
					}
				}
                num++;				
			}

			s = 0;
			float cur_val = retset[topk-1].distance;
			bool thres_flag = false;
			
		    for(int j = 0; j < n_cand1; j++){

				int a = tttest[j] % left_inv1;
				float sort_v = 1.0f * (tttest[j] - a) / left_inv0;
				float v;
				if(is_zero[a] == true) {v = -1 * sort_v;}
				else{v = sort_v;}
				
                int b = count[a];
				
				if(thres_flag == false){
				    if(s + b >= thres_pq) {thres_flag = true;}
					else {s += b;}
				}
				
				char* cur_obj1 = index_[a];

		        for(int l = 0; l < b; l++){					
					if(thres_flag == true) s++;                
					char* cur_obj = cur_obj1;
					cur_obj1 += size_per_element_;
					
					int ID = *( (int *) cur_obj);
					if (visited_array[ID] == visited_array_tag) continue;
					cur_obj += sizeof(int);
					
					float NORM = *( (float *) (cur_obj) );
				    cur_obj += sizeof(float);
					
					
					bool no_exact = false;
					float residual_NORM;
					float x = 0;
					bool is_left = true;
					float VAL = 0;

					if(sort_v  > cur_val){
						x = sort_v - cur_val;
						if( x >= delta1 * NORM ) {
							break;
						}
						
						else if(s > thres_pq && x >= delta* NORM){
							
							residual_NORM = NORM;
							if(v >= 0) is_left = false;
							cur_obj += offset00;
							goto Label2;
						}			
					}				

					int k;
					for(k = 0; k < level; k++){     
                        if(k == 0){
							if(s <= thres_pq){
							    VAL = * (float *) cur_obj;
							    cur_obj += offset00;
								residual_NORM = ( * (float *) cur_obj ) * NORM;
																
								cur_obj += offset3;
							    goto Label;								
							}
							else{
								cur_obj += offset00;
								VAL = v;
							}
						}
							
						residual_NORM =  (* (float *) cur_obj)  * NORM;
						cur_obj += sizeof(float);
						VAL += NORM * pq_dist((unsigned char*) cur_obj, table2[k], M2);
					    cur_obj += offset2;
						
Label:				    
						if(VAL < 0){
							float ttmp = -1 * VAL;
							x = ttmp - cur_val;
						}
						else{
							is_left = false;
							x = VAL - cur_val;
						}
						
						if(x <= 0){
						    break;
						}
						else{
							if(x >= delta1 * residual_NORM) {
								no_exact = true; 
							       	break;
							}
							else if(x >= delta * residual_NORM){
								break;
							}								
						}
					}
			
Label2:
					if(no_exact == false && x > 0){
						
						cur_obj -= offset1;
						int collision_  = 0;
						
						for(int jj = 0; jj < m_level; jj++){
						    collision_ += fast_count( * ( (unsigned long *) cur_obj), query_proj[jj]);
							cur_obj += sizeof(unsigned long);
						}
 						
						int y = cosine_inv * x / residual_NORM;
						if(y >= cosine_inv) y = cosine_inv - 1;
						y = cosine_table[y];

						if(is_left == true){
						   	if(collision_ >= y){
								no_exact = true;
							}
						}
						else{
							collision_ = m - collision_;
							if(collision_ >= y){
								no_exact = true;
							}
						}
					}
					
				    if(no_exact == false){
						
				        float distance = compare_ip( data[ID], query1, d) - tau;
				        if(distance < 0) distance = -1 * distance;
				        
				        visited_array[ID] = visited_array_tag;
				        Neighbor nn2;
						nn2.id = ID; nn2.distance = distance;
						if(distance >= retset[topk-1].distance) continue;
						else{
                            InsertIntoPool(retset.data(), topk, nn2);
							cur_val = retset[topk-1].distance;
						}
						
					}			
				}
				if(thres_flag == true && s < thres_pq) {s = thres_pq;}
		    }
			
            visited_list_pool_->releaseVisitedList(vl);
			
			for(int j = 0; j < topk; j++){
                search_result[i][j] = retset[j].id;			
		    }			
		}
		float time_us_per_query = stopw.getElapsedTimeMicro() / n_query;
		
		int correct = 0;
		for (int i = 0; i < n_query; i++) {
		    int* massQA2 = massQA + i * maxk; 
			for(int j = 0; j < real_topk; j++){
				bool real_flag = false;
               for(int l = 0; l < real_topk; l++){
				    if(massQA2[j] == search_result[i][l] + 1) {correct++; real_flag = true; break;}
			   }
		    } 
	    }
		float recall = 1.0f * correct / n_query / real_topk;
		cout << recall << "\t" << time_us_per_query << " us" << "\n";	
	}
	

