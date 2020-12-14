#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

using namespace cv;
using namespace std;

double get_time_sec() {
  struct timeval curr_time;
  gettimeofday(&curr_time, NULL);
  return curr_time.tv_sec + curr_time.tv_usec / 1000000.0;
}

struct hough_cmp_gt_serial
{
    hough_cmp_gt_serial(const int* _aux) : aux(_aux) {}
    inline bool operator()(int l1, int l2) const
    {
        return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
    }
    const int* aux;
};

void findLocalMaximums_serial(int numrho, int numangle, int threshold, int *accum, std::vector<int> &sort_buf, unsigned int size) {
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
		(down_index == -1 || accum[base_index] > accum[down_index]))
                sort_buf.push_back(base_index);
        }

}

std::vector<std::tuple<float, float>> hough_transform_serial(Mat img_data, int w, int h, size_t lines_max) {
  // Create the accumulator
  int accum_height = 2 * (w + h) + 1;
  int accum_width = 180;
  constexpr double PI = 3.14159;

  float cos_theta[180];
  float sin_theta[180];
  
  for (int i = 0; i < 180; i++) {
    cos_theta[i] = cos(i * PI/180.0f);
    sin_theta[i] = sin(i * PI/180.0f);
  }
  int *accum = new int[accum_height * accum_width];
  memset(accum, 0, sizeof(int) * accum_height * accum_width);

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      if (img_data.at<uint8_t>(i,j) != 0) {
        for (int theta = 0; theta < accum_width; theta++) {
          int rho = round((j * cos_theta[theta] + i * sin_theta[theta]));
          rho += (accum_height - 1)/2;
          int index = (rho * accum_width) + theta;
          accum[index]++;
        }
      }	
    }
  }

  std::vector<std::tuple<float, float>> lines;
  std::vector<int> sort_buf;
  // find local maximums
  findLocalMaximums_serial(accum_height, accum_width, 2, accum, sort_buf, accum_width*accum_height);

  // stage 3. sort the detected lines by accumulator value
  std::sort(sort_buf.begin(), sort_buf.end(), hough_cmp_gt_serial(accum));

  for (int l = 0; l < min(sort_buf.size(), lines_max); ++l) {
    int index = sort_buf.at(l);
    float rho = (index/accum_width) - ((accum_height - 1) / 2);
    float theta = (index % accum_width) * (PI / 180.0f);
    lines.push_back(std::make_tuple(rho, theta));
  }

  delete accum;
  return lines;
}


int main() {
  // Loads an image
  Mat dst, cdst;
  Mat src = imread("418/skyline.jpg", IMREAD_GRAYSCALE );

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
  std::vector<std::tuple<float, float>> lines;
  double t0 = get_time_sec();
  lines = hough_transform_serial(dst, src.cols, src.rows, 100);
  double t1 = get_time_sec();
  // Draw the lines
  for(size_t i = 0; i < lines.size(); i++) {
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
  printf("TIME: %f\n", t1 - t0);
  imwrite("out_my_serial.jpg", cdst);
  return 0;
}

