#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

#define PI 3.14159265f

using namespace cv;
using namespace std;

Mat img_data;
size_t lines_max = 100;
int accum_height;
int accum_width;
int *accum;
std::vector<std::tuple<float, float>> lines;
int *sort_buf;
size_t buf_idx;
int threads;

double get_time_sec() {
  struct timeval curr_time;
  gettimeofday(&curr_time, NULL);
  return curr_time.tv_sec + curr_time.tv_usec / 1000000.0;
}

struct hough_cmp_gt
{
  hough_cmp_gt(const int* _aux) : aux(_aux) {}
  inline bool operator()(int l1, int l2) const
  {
    return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
  }
  const int* aux;
};

void hough_transform(int w, int h) {
  #pragma omp parallel num_threads(threads)
  {
  
  #pragma omp for collapse(2)
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      if (img_data.at<uint8_t>(i,j) != 0) {
        for (int theta = 0; theta < accum_width; theta++) {
	  int rho = round(j * cos(theta * (PI / 180.0f)) + i * sin(theta * (PI / 180.0f)));
	  rho += (accum_height - 1)/2;
	  int index = (rho * accum_width) + theta;
          #pragma omp atomic
	  accum[index]++;
	}
      }	
    }
  }

  // find local maximums
  int numrho = accum_height;
  int numangle = accum_width;
  int threshold = 2;
  int idx;
  
  #pragma omp for collapse(2)
  for(int r = 0; r < numrho; r++ ) {
    for(int n = 0; n < numangle; n++ ) {
      int base_index = r * (numangle) + n;
      int left_index = (base_index - 1) < 0 ? -1 : base_index - 1;
      int right_index = (base_index + 1) > (numrho*numangle) ? -1 : base_index - 1;
      int up_index = (r > 0) ? (r-1) * (numangle) + n: -1;
      int down_index = (r < numrho - 1) ? (r+1) * (numangle) + n: -1;
      if (accum[base_index] >= threshold &&
        (left_index == -1 || accum[base_index] > accum[left_index]) && 
  	(right_index == -1 || accum[base_index] >= accum[right_index]) &&
        (up_index == -1 || accum[base_index] >= accum[up_index]) && 
        (down_index == -1 || accum[base_index] > accum[down_index])){
          #pragma omp atomic capture
          idx = buf_idx++;
          sort_buf[idx] = base_index;
      }
    }
  }

  }
  // stage 3. sort the detected lines by accumulator value
  std::sort(sort_buf, sort_buf + buf_idx, hough_cmp_gt(accum));

  for (int l = 0; l < min(buf_idx, lines_max); ++l) {
    int index = sort_buf[l];
    float rho = (index/accum_width) - ((accum_height - 1) / 2);
    float theta = (index % accum_width) * (PI / 180.0f);
    lines.push_back(std::make_tuple(rho, theta));
  }
}

void spawn_threads(Mat dst, Mat src) {
  img_data = dst;
  accum_height = 2 * (src.cols + src.rows) + 1;
  accum_width = 180;

  accum = new int[accum_height * accum_width]();
  //memset(accum, 0, sizeof(int) * accum_height * accum_width);

  sort_buf = new int[accum_height  * accum_width]();
  buf_idx = 0;

  int w = src.cols;
  int h = src.rows;
  hough_transform(w, h);
}

int main(int argc, char **argv) {
  //double start = get_time_sec();
  if(argc != 4) {
      printf("Please supply the proper arguments: ./openmp.0 -p [numProcessors] [inputFile]\n");
      return -1;
  }
  if (strcmp(argv[1], "-p") != 0) {
      printf("Use -p to pass in the number of processors\n");
      return -1;
  }
  threads = atoi(argv[2]);

  // Loads an image
  Mat dst, cdst;
  Mat src = imread(argv[3], IMREAD_GRAYSCALE );

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
  vector<Vec2f> cvlines; // will hold the results of the detection
  HoughLines(dst, cvlines, 1, CV_PI/180, 150, 0, 0); // runs the actual detection

  double t0 = get_time_sec();
  spawn_threads(dst, src);
  double t1 = get_time_sec();
 
  // Draw the lines
  for( size_t i = 0; i < lines.size(); i++ ) {
    float rho = get<0>(lines[i]), theta = get<1>(lines[i]);	
    float crho = cvlines[i][0], ctheta = cvlines[i][1];	
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + 2000*(-b));
    pt1.y = cvRound(y0 + 2000*(a));
    pt2.x = cvRound(x0 - 2000*(-b));
    pt2.y = cvRound(y0 - 2000*(a));
    line(cdst, pt1, pt2, Scalar(0,0,255), 3, 16);
  }
  printf("Time: %f\n", t1 - t0);
  imwrite("out_openmp.jpg", cdst);
  return 0;
}


