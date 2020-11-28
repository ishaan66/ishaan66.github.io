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

#define PI 3.14159265

using namespace cv;
using namespace std;


struct hough_cmp_gt
{
    hough_cmp_gt(const int64_t* _aux) : aux(_aux) {}
    inline bool operator()(int64_t l1, int64_t l2) const
    {
        return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
    }
    const int64_t* aux;
};

void findLocalMaximums(int numrho, int numangle, int threshold, int64_t *accum, std::vector<int> &sort_buf) {
    for(int r = 0; r < numrho; r++ )
        for(int n = 0; n < numangle; n++ )
        {
            int base_index = r * (numangle) + n;
	    int up_index = (r > 0) ? (r-1) * (numangle) + n: -1;
	    int down_index = (r < numrho - 1) ? (r+1) * (numangle) + n: -1;
            if( accum[base_index] >= threshold &&
                accum[base_index] > accum[base_index - 1] && accum[base_index] >= accum[base_index + 1] &&
                (up_index == -1 || accum[base_index] >= accum[up_index]) && 
		(down_index == -1 || accum[base_index] > accum[down_index]) )
                sort_buf.push_back(base_index);
        }

}

std::vector<std::tuple<int, int>> hough_transform(Mat img_data, int w, int h) {
  // Create the accumulator
  int accum_height = 2 * (int) sqrt(w*w + h*h);
  int accum_width = 180;

  int64_t accum[accum_height * accum_width];
  memset(accum, 0, sizeof(int64_t) * accum_height * accum_width);

  double cos_theta[180];
  double sin_theta[180];
  int64_t theta_vals[180];
  
  for (int i = 0; i < 180; i++) {
    cos_theta[i] = cos(i * PI/180);
    sin_theta[i] = sin(i * PI/180);
    theta_vals[i] = i;
  }

  __m256d x;
  __m256d y;

  __m256d rho_vec;
  __m256d cos_vec1;
  __m256d sin_vec1;
  __m256i theta_vec1;
  __m256d rho_floor;
  __m256i rho_index;
  __m256i values;


  int64_t accum_w = accum_width;
  double  half_rho_height = (accum_height-1)/2;
  __m256d half_rho_height_v = _mm256_set_pd(half_rho_height, half_rho_height, half_rho_height, half_rho_height);
  __m256i accum_width_v = _mm256_set_epi64x(accum_w, accum_w, accum_w, accum_w);

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
	//if (img_data[i*w + j] != 0) {
	if (img_data.at<float>(i, j) != 0) {
	  double xval = j;
	  double yval = i;

	  x = _mm256_broadcast_sd(&xval);
	  y = _mm256_broadcast_sd(&yval);

	  int index_buf[4];
	  int value_buf[4];

	  for (int64_t theta = 0; theta < accum_width; theta += 4) {
	    rho_vec = _mm256_setzero_pd();
	    cos_vec1 = _mm256_loadu_pd(&cos_theta[theta]);
	    sin_vec1 = _mm256_loadu_pd(&sin_theta[theta]);
	    theta_vec1 = _mm256_set_epi64x(theta, theta+1, theta+2, theta+3);

	    rho_vec = _mm256_fmadd_pd(x, cos_vec1, rho_vec);
	    rho_vec = _mm256_fmadd_pd(y, sin_vec1, rho_vec);

	    rho_floor = _mm256_round_pd(rho_vec, 0x09);
	    //rho_index = _mm256_castpd_si256(rho_floor);
            
	    rho_index = _mm256_castpd_si256(_mm256_add_pd(half_rho_height_v, rho_floor));
            rho_index = _mm256_castpd_si256(_mm256_fmadd_pd(_mm256_castsi256_pd(accum_width_v), 
			    _mm256_castsi256_pd(rho_index), _mm256_castsi256_pd(theta_vec1)));


	    accum[_mm256_extract_epi64(rho_index, 0)]++;
	    accum[_mm256_extract_epi64(rho_index, 1)]++;
	    accum[_mm256_extract_epi64(rho_index, 2)]++;
	    accum[_mm256_extract_epi64(rho_index, 3)]++;
	  }
	}	
    }
  }

  std::vector<std::tuple<int, int>> lines;
  std::vector<int> sort_buf;
  // find local maximums
  findLocalMaximums(accum_height, accum_width, 2, accum, sort_buf);

  // stage 3. sort the detected lines by accumulator value
  std::sort(sort_buf.begin(), sort_buf.end(), hough_cmp_gt(accum));

  for (int index: sort_buf) {
    int rho = index/accum_width;
    int theta = index % accum_width;
    std::cout << "rho: " << rho << ", theta: " << theta << std::endl;
    lines.push_back(std::make_tuple(rho, theta));
  }

  
  for (int i = 0;i < accum_height; i++) {
    for (int j = 0; j < accum_width; j++) {
      std::cout << accum[i * accum_width + j] << " ";
    }
    std::cout << "\n";
  }
  
  
  
  std::cout << "accum size is " << accum_height * accum_width  << "\n";
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
    //vector<Vec2f> lines; // will hold the results of the detection
    std::vector<std::tuple<int, int>> lines;
    //HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection
    lines = hough_transform(dst, src.rows, src.cols);
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        //float rho = lines[i][0], theta = lines[i][1];
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

    // Probabilistic Line Transform
    //vector<Vec4i> linesP; // will hold the results of the detection
    //HoughLinesP(dst, linesP, 1, CV_PI/180, 50, 50, 10 ); // runs the actual detection
    //![hough_lines_p]
    //![draw_lines_p]
    // Draw the lines
    /*
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, 16);
    }
    */
    imwrite("out_simd.jpg", cdst);

  /**
  Mat dst;
  int h = 10;
  int w = 10;

  // Loads an image
  Mat src = imread("nyc.jpg" , IMREAD_GRAYSCALE );

  // Check if image is loaded fine
  if(src.empty()){
      printf(" Error opening image\n");
      return -1;
  }

  // Edge detection
  Canny(src, dst, 50, 200, 3);

  // Standard Hough Line Transform
  vector<Vec2f> lines; // will hold the results of the detection
  HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection
  
  unsigned char *img_data = new unsigned char[h*w];
  img_data[50] = 1;
  img_data[60] = 1;
  img_data[22] = 1; 
  std::vector<std::tuple<int,int>> lines = hough_transform(img_data, w, h);
  
  for (auto elem: lines) {
    cout << get<0>(elem) << " " << get<1>(elem)  << endl;	  
  }
  */
  
  return 0;
}
