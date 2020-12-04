#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <stdio.h>
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include <inttypes.h>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265f

using namespace cv;
using namespace std;

static __inline__ unsigned long long rdtsc(void) {
	unsigned hi, lo;
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	return ((unsigned long long)lo) | (((unsigned long long)hi)<<32);
}

struct hough_cmp_gt
{
    hough_cmp_gt(const int32_t* _aux) : aux(_aux) {}
    inline bool operator()(int32_t l1, int32_t l2) const
    {
        return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
    }
    const int32_t* aux;
};

void findLocalMaximums(int numrho, int numangle, int threshold, int32_t *accum, std::vector<int> &sort_buf, unsigned int size) {
    for(int r = 0; r < numrho; r++ )
        for(int n = 0; n < numangle; n++ )
        {
            int base_index = r * (numangle) + n;
	    int left_index = (base_index - 1) < 0 ? -1 : base_index - 1;
	    int right_index = (base_index + 1) > size ? -1 : base_index - 1;
	    int up_index = (r > 0) ? (r-1) * (numangle) + n: -1;
	    int down_index = (r < numrho - 1) ? (r+1) * (numangle) + n: -1;
            if (accum[base_index] >= threshold &&
                (left_index == -1 || accum[base_index] > accum[left_index]) && 
		(right_index == -1 || accum[base_index] >= accum[right_index]) &&
                (up_index == -1 || accum[base_index] >= accum[up_index]) && 
		(down_index == -1 || accum[base_index] > accum[down_index])) {
                sort_buf.push_back(base_index);
	    }
        }

}

std::vector<std::tuple<float, float>> hough_transform(Mat img_data, int w, int h, size_t lines_max) {
  // Create the accumulator
  int32_t accum_height = 2 * (w + h) + 1;
  int32_t accum_width = 180;

  int32_t accum[accum_height * accum_width];
  memset(accum, 0, sizeof(int32_t) * accum_height * accum_width);

  float cos_theta[180];
  float sin_theta[180];
  
  for (int i = 0; i < 180; i++) {
    cos_theta[i] = cos(i * PI/180.0f);
    sin_theta[i] = sin(i * PI/180.0f);
  }

  __m256 x_y_1;
  __m256 x_y_2;
  __m256 x_y_3;
  //__m256 x_y_4;
  //__m256 x_y_5;

  __m256 rho_vec_set1_1;
  __m256 rho_vec_set1_2;

  __m256 rho_vec_set2_1;
  __m256 rho_vec_set2_2;

  __m256 rho_vec_set3_1;
  __m256 rho_vec_set3_2;

  __m256 rho_vec_set4_1;
  __m256 rho_vec_set4_2;

  __m256 rho_vec_set5_1;
  __m256 rho_vec_set5_2;

  __m256 trig_vec1;
  __m256 trig_vec2;

  __m256 theta_vec1;
  __m256 theta_vec2;

  __m256i index1;
  __m256 theta_offset;

  float half_rho_height = (accum_height-1)/2;
  float accum_w = accum_width * 1.0f;
  __m256 half_rho_height_v = _mm256_broadcast_ss(&half_rho_height);
  __m256 accum_width_v = _mm256_broadcast_ss((float *)&accum_w);

  std::vector<std::tuple<float, float>> positions;
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      if (img_data.at<uint8_t>(i, j) != 0) {
        positions.push_back(std::make_tuple(i*1.0f,j*1.0f));
      }
    }
  }


  float temp_theta[16] = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,
	                  8.0f,9.0f,10.0f,11.0f,12.0f,12.0f,14.0f,15.0f};
  std::cout << "Size of positions: " << positions.size() << std::endl;
  auto t0 = rdtsc();
  for (int i = 0; i < positions.size()-4; i+=5) { // off by one error
    float x_val_1 = std::get<0>(positions.at(i));
    float y_val_1 = std::get<1>(positions.at(i));
    
    float x_val_2 = std::get<0>(positions.at(i+1));
    float y_val_2 = std::get<1>(positions.at(i+1));
    
    float x_val_3 = std::get<0>(positions.at(i+2));
    float y_val_3 = std::get<1>(positions.at(i+2));
    
    float x_val_4 = std::get<0>(positions.at(i+3));
    float y_val_4 = std::get<1>(positions.at(i+3));

    float x_val_5 = std::get<0>(positions.at(i+4));
    float y_val_5 = std::get<1>(positions.at(i+4));

    float offset = 16.0f;
    theta_offset = _mm256_broadcast_ss((float *)&offset);
    theta_vec1 = _mm256_loadu_ps(temp_theta);
    theta_vec2 = _mm256_loadu_ps(temp_theta+8);

    for (int32_t theta = 0; theta < accum_width; theta += 16) {

      rho_vec_set1_1 = _mm256_setzero_ps();
      rho_vec_set1_2 = _mm256_setzero_ps();

      rho_vec_set2_1 = _mm256_setzero_ps();
      rho_vec_set2_2 = _mm256_setzero_ps();

      rho_vec_set3_1 = _mm256_setzero_ps();
      rho_vec_set3_2 = _mm256_setzero_ps();

      rho_vec_set4_1 = _mm256_setzero_ps();
      rho_vec_set4_2 = _mm256_setzero_ps();

      rho_vec_set5_1 = _mm256_setzero_ps();
      rho_vec_set5_2 = _mm256_setzero_ps();

      trig_vec1 = _mm256_loadu_ps(&(cos_theta[theta]));
      trig_vec2 = _mm256_loadu_ps(&(cos_theta[theta+8]));

      x_y_1 = _mm256_broadcast_ss((float *)&x_val_1);
      x_y_2 = _mm256_broadcast_ss((float *)&x_val_2);
      x_y_3 = _mm256_broadcast_ss((float *)&x_val_3);
      //x_y_4 = _mm256_broadcast_ss((float *)&x_val_4);
      //x_y_5 = _mm256_broadcast_ss((float *)&x_val_5);

      rho_vec_set1_1 = _mm256_fmadd_ps(x_y_1, trig_vec1, rho_vec_set1_1);
      rho_vec_set1_2 = _mm256_fmadd_ps(x_y_1, trig_vec2, rho_vec_set1_2);

      rho_vec_set2_1 = _mm256_fmadd_ps(x_y_2, trig_vec1, rho_vec_set2_1);
      rho_vec_set2_2 = _mm256_fmadd_ps(x_y_2, trig_vec2, rho_vec_set2_2);

      rho_vec_set3_1 = _mm256_fmadd_ps(x_y_3, trig_vec1, rho_vec_set3_1);
      rho_vec_set3_2 = _mm256_fmadd_ps(x_y_3, trig_vec2, rho_vec_set3_2);

      x_y_1 = _mm256_broadcast_ss((float *)&x_val_4);
      rho_vec_set4_1 = _mm256_fmadd_ps(x_y_1, trig_vec1, rho_vec_set4_1);
      rho_vec_set4_2 = _mm256_fmadd_ps(x_y_1, trig_vec2, rho_vec_set4_2);
      //rho_vec_set3_3 = _mm256_fmadd_ps(x_y_3, trig_vec3, rho_vec_set3_3);
      x_y_2 = _mm256_broadcast_ss((float *)&x_val_5);
      rho_vec_set5_1 = _mm256_fmadd_ps(x_y_2, trig_vec1, rho_vec_set5_1);
      rho_vec_set5_2 = _mm256_fmadd_ps(x_y_2, trig_vec2, rho_vec_set5_2);

      // Using FMA - need at least 10 independent calls to fma
      trig_vec1 = _mm256_loadu_ps(&(sin_theta[theta]));
      trig_vec2 = _mm256_loadu_ps(&(sin_theta[theta+8]));
      //trig_vec3 = _mm256_loadu_ps(&(sin_theta[theta+16]));

      x_y_1 = _mm256_broadcast_ss((float *)&y_val_1);
      x_y_2 = _mm256_broadcast_ss((float *)&y_val_2);
      x_y_3 = _mm256_broadcast_ss((float *)&y_val_3);
      //x_y_4 = _mm256_broadcast_ss((float *)&y_val_4);
      //x_y_5 = _mm256_broadcast_ss((float *)&y_val_5);
      
      rho_vec_set1_1 = _mm256_fmadd_ps(x_y_1, trig_vec1, rho_vec_set1_1);
      rho_vec_set1_2 = _mm256_fmadd_ps(x_y_1, trig_vec2, rho_vec_set1_2);

      rho_vec_set2_1 = _mm256_fmadd_ps(x_y_2, trig_vec1, rho_vec_set2_1);
      rho_vec_set2_2 = _mm256_fmadd_ps(x_y_2, trig_vec2, rho_vec_set2_2);

      rho_vec_set3_1 = _mm256_fmadd_ps(x_y_3, trig_vec1, rho_vec_set3_1);
      rho_vec_set3_2 = _mm256_fmadd_ps(x_y_3, trig_vec2, rho_vec_set3_2);

      x_y_1 = _mm256_broadcast_ss((float *)&y_val_4);
      rho_vec_set4_1 = _mm256_fmadd_ps(x_y_1, trig_vec1, rho_vec_set4_1);
      rho_vec_set4_2 = _mm256_fmadd_ps(x_y_1, trig_vec2, rho_vec_set4_2);

      x_y_2 = _mm256_broadcast_ss((float *)&y_val_5);
      rho_vec_set5_1 = _mm256_fmadd_ps(x_y_2, trig_vec1, rho_vec_set5_1);
      rho_vec_set5_2 = _mm256_fmadd_ps(x_y_2, trig_vec2, rho_vec_set5_2);

      //ROUNDING THE RHO VALUES//
      rho_vec_set1_1 = _mm256_round_ps(rho_vec_set1_1, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
      rho_vec_set1_2 = _mm256_round_ps(rho_vec_set1_2, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

      rho_vec_set2_1 = _mm256_round_ps(rho_vec_set2_1, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
      rho_vec_set2_2 = _mm256_round_ps(rho_vec_set2_2, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

      rho_vec_set3_1 = _mm256_round_ps(rho_vec_set3_1, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
      rho_vec_set3_2 = _mm256_round_ps(rho_vec_set3_2, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

      rho_vec_set4_1 = _mm256_round_ps(rho_vec_set4_1, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
      rho_vec_set4_2 = _mm256_round_ps(rho_vec_set4_2, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

      rho_vec_set5_1 = _mm256_round_ps(rho_vec_set5_1, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
      rho_vec_set5_2 = _mm256_round_ps(rho_vec_set5_2, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

      //PARTIAL INDEX COMPUTATION//
      rho_vec_set1_1 = _mm256_add_ps(half_rho_height_v, rho_vec_set1_1);
      rho_vec_set1_2 = _mm256_add_ps(half_rho_height_v, rho_vec_set1_2);

      rho_vec_set2_1 = _mm256_add_ps(half_rho_height_v, rho_vec_set2_1);
      rho_vec_set2_2 = _mm256_add_ps(half_rho_height_v, rho_vec_set2_2);

      rho_vec_set3_1 = _mm256_add_ps(half_rho_height_v, rho_vec_set3_1);
      rho_vec_set3_2 = _mm256_add_ps(half_rho_height_v, rho_vec_set3_2);

      rho_vec_set4_1 = _mm256_add_ps(half_rho_height_v, rho_vec_set4_1);
      rho_vec_set4_2 = _mm256_add_ps(half_rho_height_v, rho_vec_set4_2);

      rho_vec_set5_1 = _mm256_add_ps(half_rho_height_v, rho_vec_set5_1);
      rho_vec_set5_2 = _mm256_add_ps(half_rho_height_v, rho_vec_set5_2);
      

      if (theta != 0) {
        theta_vec1 = _mm256_add_ps(theta_offset, theta_vec1);
        theta_vec2 = _mm256_add_ps(theta_offset, theta_vec2);
      }
      /*
      theta_vec1 = _mm256_set_ps(theta+7.0f, theta+6.0f, theta+5.0f, theta+4.0f,
    			         theta+3.0f, theta+2.0f, theta+1.0f, theta+0.0f);
      theta_vec2 = _mm256_set_ps(theta+15.0f, theta+14.0f, theta+13.0f, theta+12.0f,
			         theta+11.0f, theta+10.0f, theta+9.0f, theta+8.0f);
      */
      index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set1_1, theta_vec1));

      accum[_mm256_extract_epi32(index1, 0)]++;
      accum[_mm256_extract_epi32(index1, 1)]++;
      accum[_mm256_extract_epi32(index1, 2)]++;
      accum[_mm256_extract_epi32(index1, 3)]++;
      if (theta < 176) {
        accum[_mm256_extract_epi32(index1, 4)]++;
        accum[_mm256_extract_epi32(index1, 5)]++;
        accum[_mm256_extract_epi32(index1, 6)]++;
        accum[_mm256_extract_epi32(index1, 7)]++;

        index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set1_2, theta_vec2));
        accum[_mm256_extract_epi32(index1, 0)]++;
        accum[_mm256_extract_epi32(index1, 1)]++;
        accum[_mm256_extract_epi32(index1, 2)]++;
        accum[_mm256_extract_epi32(index1, 3)]++;
	accum[_mm256_extract_epi32(index1, 4)]++;
	accum[_mm256_extract_epi32(index1, 5)]++;
	accum[_mm256_extract_epi32(index1, 6)]++;
	accum[_mm256_extract_epi32(index1, 7)]++;
/*
	index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set1_3, theta_vec3));
	accum[_mm256_extract_epi32(index1, 0)]++;
	accum[_mm256_extract_epi32(index1, 1)]++;
	accum[_mm256_extract_epi32(index1, 2)]++;
	accum[_mm256_extract_epi32(index1, 3)]++;
	accum[_mm256_extract_epi32(index1, 4)]++;
	accum[_mm256_extract_epi32(index1, 5)]++;
	accum[_mm256_extract_epi32(index1, 6)]++;
	accum[_mm256_extract_epi32(index1, 7)]++;
*/
      } 

      index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set2_1, theta_vec1));

      accum[_mm256_extract_epi32(index1, 0)]++;
      accum[_mm256_extract_epi32(index1, 1)]++;
      accum[_mm256_extract_epi32(index1, 2)]++;
      accum[_mm256_extract_epi32(index1, 3)]++;
      if (theta < 176) {
        accum[_mm256_extract_epi32(index1, 4)]++;
        accum[_mm256_extract_epi32(index1, 5)]++;
        accum[_mm256_extract_epi32(index1, 6)]++;
        accum[_mm256_extract_epi32(index1, 7)]++;

        index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set2_2, theta_vec2));
        accum[_mm256_extract_epi32(index1, 0)]++;
        accum[_mm256_extract_epi32(index1, 1)]++;
        accum[_mm256_extract_epi32(index1, 2)]++;
        accum[_mm256_extract_epi32(index1, 3)]++;
	accum[_mm256_extract_epi32(index1, 4)]++;
	accum[_mm256_extract_epi32(index1, 5)]++;
	accum[_mm256_extract_epi32(index1, 6)]++;
	accum[_mm256_extract_epi32(index1, 7)]++;
/*
	index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set2_3, theta_vec3));
	accum[_mm256_extract_epi32(index1, 0)]++;
	accum[_mm256_extract_epi32(index1, 1)]++;
	accum[_mm256_extract_epi32(index1, 2)]++;
	accum[_mm256_extract_epi32(index1, 3)]++;
	accum[_mm256_extract_epi32(index1, 4)]++;
	accum[_mm256_extract_epi32(index1, 5)]++;
	accum[_mm256_extract_epi32(index1, 6)]++;
	accum[_mm256_extract_epi32(index1, 7)]++;
**/
      }       
      

      index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set3_1, theta_vec1));

      accum[_mm256_extract_epi32(index1, 0)]++;
      accum[_mm256_extract_epi32(index1, 1)]++;
      accum[_mm256_extract_epi32(index1, 2)]++;
      accum[_mm256_extract_epi32(index1, 3)]++;
      if (theta < 176) {
        accum[_mm256_extract_epi32(index1, 4)]++;
        accum[_mm256_extract_epi32(index1, 5)]++;
        accum[_mm256_extract_epi32(index1, 6)]++;
        accum[_mm256_extract_epi32(index1, 7)]++;

        index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set3_2, theta_vec2));
        accum[_mm256_extract_epi32(index1, 0)]++;
        accum[_mm256_extract_epi32(index1, 1)]++;
        accum[_mm256_extract_epi32(index1, 2)]++;
        accum[_mm256_extract_epi32(index1, 3)]++;
	accum[_mm256_extract_epi32(index1, 4)]++;
	accum[_mm256_extract_epi32(index1, 5)]++;
	accum[_mm256_extract_epi32(index1, 6)]++;
	accum[_mm256_extract_epi32(index1, 7)]++;
/*
	index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set3_3, theta_vec3));
	accum[_mm256_extract_epi32(index1, 0)]++;
	accum[_mm256_extract_epi32(index1, 1)]++;
	accum[_mm256_extract_epi32(index1, 2)]++;
	accum[_mm256_extract_epi32(index1, 3)]++;
	accum[_mm256_extract_epi32(index1, 4)]++;
	accum[_mm256_extract_epi32(index1, 5)]++;
	accum[_mm256_extract_epi32(index1, 6)]++;
	accum[_mm256_extract_epi32(index1, 7)]++;
**/
      } 
      index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set4_1, theta_vec1));

      accum[_mm256_extract_epi32(index1, 0)]++;
      accum[_mm256_extract_epi32(index1, 1)]++;
      accum[_mm256_extract_epi32(index1, 2)]++;
      accum[_mm256_extract_epi32(index1, 3)]++;
      if (theta < 176) {
        accum[_mm256_extract_epi32(index1, 4)]++;
        accum[_mm256_extract_epi32(index1, 5)]++;
        accum[_mm256_extract_epi32(index1, 6)]++;
        accum[_mm256_extract_epi32(index1, 7)]++;

        index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set4_2, theta_vec2));
        accum[_mm256_extract_epi32(index1, 0)]++;
        accum[_mm256_extract_epi32(index1, 1)]++;
        accum[_mm256_extract_epi32(index1, 2)]++;
        accum[_mm256_extract_epi32(index1, 3)]++;
	accum[_mm256_extract_epi32(index1, 4)]++;
	accum[_mm256_extract_epi32(index1, 5)]++;
	accum[_mm256_extract_epi32(index1, 6)]++;
	accum[_mm256_extract_epi32(index1, 7)]++;

      }	
      index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set5_1, theta_vec1));

      accum[_mm256_extract_epi32(index1, 0)]++;
      accum[_mm256_extract_epi32(index1, 1)]++;
      accum[_mm256_extract_epi32(index1, 2)]++;
      accum[_mm256_extract_epi32(index1, 3)]++;
      if (theta < 176) {
        accum[_mm256_extract_epi32(index1, 4)]++;
        accum[_mm256_extract_epi32(index1, 5)]++;
        accum[_mm256_extract_epi32(index1, 6)]++;
        accum[_mm256_extract_epi32(index1, 7)]++;

        index1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(accum_width_v, rho_vec_set5_2, theta_vec2));
        accum[_mm256_extract_epi32(index1, 0)]++;
        accum[_mm256_extract_epi32(index1, 1)]++;
        accum[_mm256_extract_epi32(index1, 2)]++;
        accum[_mm256_extract_epi32(index1, 3)]++;
	accum[_mm256_extract_epi32(index1, 4)]++;
	accum[_mm256_extract_epi32(index1, 5)]++;
	accum[_mm256_extract_epi32(index1, 6)]++;
	accum[_mm256_extract_epi32(index1, 7)]++;
      }
    }
  } 
  auto t1 = rdtsc();
  std::cout << "Performance (kernel actual): " << (((w*h)*180*5)/(double)(t1 - t0)) <<std::endl;


  std::vector<std::tuple<float, float>> lines;
  std::vector<int> sort_buf;
  // find local maximums
  findLocalMaximums(accum_height, accum_width, 2, accum, sort_buf, accum_height*accum_width);

  // stage 3. sort the detected lines by accumulator value
  std::sort(sort_buf.begin(), sort_buf.end(), hough_cmp_gt(accum));

  for (int l = 0; l < min(sort_buf.size(), lines_max); ++l) {
    int index = sort_buf.at(l);
    float rho = index/accum_width - half_rho_height;
    float theta = (index % accum_width) * (PI / 180.0f);
    lines.push_back(std::make_tuple(rho, theta));
  }

  return lines;
}

int main() {
    // Loads an image
    Mat dst, cdst;
    Mat src = imread("grid.jpg", IMREAD_GRAYSCALE );

    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
        return -1;
    }

    // Edge detection
    Canny(src, dst, 50, 200, 3);

    // Copy edges to the images that will display the results in BGR
    cvtColor(dst, cdst, COLOR_GRAY2BGR);

    // Standard Hough Line Transform
    std::vector<std::tuple<float, float>> lines;
    //HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection
    unsigned long long t1 = rdtsc();
    lines = hough_transform(dst, src.cols, src.rows, 100);
    //lines = hough_transform(dst, 100, 100, 100);
    unsigned long long t2 = rdtsc();
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
	float rho = get<0>(lines[i]), theta = get<1>(lines[i]);	
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line(cdst, pt1, pt2, Scalar(0,0,255), 3, 16);
    }
    imwrite("out_simd.png", cdst);
    std::cout << "Number of cycles (kernel): " << (t2 - t1) <<std::endl;
    std::cout << "Performance (kernel): " << (((src.cols*src.rows)*180*4)/(double)(t2 - t1)) <<std::endl;

  return 0;
}
