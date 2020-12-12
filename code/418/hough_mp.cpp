#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <fstream>
#include <string>
#define PI 3.14159265f

using namespace cv;
using namespace std;

Mat img_data;
size_t lines_max = 100;
int accum_height;
int accum_width;
int *accum_global;
int *sort_buf;
size_t buf_idx;
int threads;
float half_rho_height;
float cos_theta[180];
float sin_theta[180];  

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

std::vector<std::tuple<float, float>> hough_transform(int w, int h) {
  #pragma omp parallel num_threads(threads)
  {
  int32_t accum[accum_height * accum_width];
  memset(accum, 0, sizeof(int32_t) * accum_height * accum_width);

  #pragma omp for collapse(2) schedule (dynamic, (h*w)/threads)
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      if (img_data.at<uint8_t>(i,j) != 0) {
        for (int theta = 0; theta < accum_width; theta++) {
	  int rho = round(j * cos_theta[theta] + i * sin_theta[theta]);
	  rho += (accum_height - 1)/2;
	  int index = (rho * accum_width) + theta;
          #pragma omp atomic
	  accum[index]++;
	}
      }	
    }
  }
  #pragma omp critical (update)
  {
  for (int i = 0; i < accum_height; i++) {
    for (int j = 0; j < accum_width; j++) {
      int index = (i * accum_width) + j;
      //#pragma omp atomic
      accum_global[index] += accum[index];
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
      if (accum_global[base_index] >= threshold &&
        (left_index == -1 || accum_global[base_index] > accum_global[left_index]) && 
  	(right_index == -1 || accum_global[base_index] >= accum_global[right_index]) &&
        (up_index == -1 || accum_global[base_index] >= accum_global[up_index]) && 
        (down_index == -1 || accum_global[base_index] > accum_global[down_index])){
        #pragma omp atomic capture
        idx = buf_idx++;
        sort_buf[idx] = base_index;
      }
    }
  }

  }
  // stage 3. sort the detected lines by accumulator value
  std::sort(sort_buf, sort_buf + buf_idx, hough_cmp_gt(accum_global));
  std::vector<std::tuple<float, float>> lines;

  for (int l = 0; l < min(buf_idx, lines_max); ++l) {
    int index = sort_buf[l];
    float rho = (index/accum_width) - half_rho_height;
    float theta = (index % accum_width) * (PI / 180.0f);
    lines.push_back(std::make_tuple(rho, theta));
  }
  return lines;
}

std::vector<std::tuple<float, float>> spawn_threads(Mat dst, Mat src) {
  img_data = dst;
  accum_height = 2 * (src.cols + src.rows) + 1;
  accum_width = 180;

  accum_global = new int[accum_height * accum_width]();
  sort_buf = new int[accum_height  * accum_width]();
  buf_idx = 0;
  half_rho_height = (accum_height-1)/2;

  for (int i = 0; i < 180; i++) {
    cos_theta[i] = cos(i * PI/180.0f);
    sin_theta[i] = sin(i * PI/180.0f);
  }
  int w = src.cols;
  int h = src.rows;
  return hough_transform(w, h);
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
  std::vector<std::tuple<float, float>> data;

  // Check if image is loaded fine
  if(src.empty()){
    printf(" Error opening image\n");
    return -1;
  }

  // Edge detection
  Canny(src, dst, 50, 200, 3);

  // Copy edges to the images that will display the results in BGR
  cvtColor(dst, cdst, COLOR_GRAY2BGR);
 
  int rows = 2520;
  int cols = 2520;
  dst = Mat(rows,cols, CV_64F, double(1));
  std::vector<std::tuple<float, float>> lines;

  // Standard Hough Line Transform

  for (int t = 1; t <= 28; t++) {
    std::cout << "iter: " << t << std::endl;
    threads = t;
    double t0 = get_time_sec();
    spawn_threads(dst, dst);
    spawn_threads(dst, dst);
    spawn_threads(dst, dst);
    spawn_threads(dst, dst);
    lines = spawn_threads(dst, dst);
    double t1 = get_time_sec();
    std::cout << " time: " << (t1-t0)/5 << std::endl;
    data.push_back(std::make_tuple(t, (t1 - t0)/5));
  }
  // Draw the lines
  for( size_t i = 0; i < lines.size(); i++ ) {
    float rho = get<0>(lines[i]), theta = get<1>(lines[i]);	
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + 2000*(-b));
    pt1.y = cvRound(y0 + 2000*(a));
    pt2.x = cvRound(x0 - 2000*(-b));
    pt2.y = cvRound(y0 - 2000*(a));
    line(cdst, pt1, pt2, Scalar(0,0,255), 3, 16);
  }
  //printf("Time: %f\n", t1 - t0);
  imwrite("out_openmp.jpg", cdst);

  std::string outfile = "execution_time_mp.csv";
  std::ofstream myfile(outfile);

  if (myfile.fail()) {
    cerr << "Error opening file\n" << std::endl;
	return -1;
  }

  for (std::tuple<float, float> tup: data) {
    float size = get<0>(tup);
    float cycles = get<1>(tup);
    myfile << size << ", " << cycles << std::endl;
  }
  myfile.close();

  return 0;
}


