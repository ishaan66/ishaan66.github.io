#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <tuple>
#include <../opencv2-master/include/opencv2>
#include <algorithm>

using namespace cv;

struct hough_cmp_gt
{
    hough_cmp_gt(const int* _aux) : aux(_aux) {}
    inline bool operator()(int l1, int l2) const
    {
        return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
    }
    const int* aux;
};

void findLocalMaximums(int numrho, int numangle, int threshold, int *accum, std::vector<int> &sort_buf) {
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

std::vector<std::tuple<int, int>> hough_transform(unsigned char *img_data, int w, int h) {
  // Create the accumulator
  int accum_height = 2 * (int) sqrt(w*w + h*h);
  int accum_width = 180;

  int *accum = new int[accum_height * accum_width];
  memset(accum, 0, sizeof(int) * accum_height * accum_width);

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
	if (img_data[i*w + j] != 0) {
	  for (int theta = 0; theta < accum_width; theta++) {
	    int rho = round(j * cos(theta) + i * sin(theta));
	    rho += (accum_height - 1)/2;
	    int index = (rho * accum_width) + theta;
	    //std::cout << "index is " << index;
	    accum[index]++;
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

  /**
  for (int i = 0;i < accum_height; i++) {
    for (int j = 0; j < accum_width; j++) {
      std::cout << accum[i * accum_width + j] << " ";
    }
    std::cout << "\n";
  }
  */
  
  
  std::cout << "accum size is " << accum_height * accum_width  << "\n";
  delete accum;
  return lines;
}


int main() {
  int h = 10;
  int w = 10;

  Mat dst;

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

  /**
  unsigned char *img_data = new unsigned char[h*w];
  img_data[50] = 1;
  img_data[60] = 1;
  img_data[22] = 1; 
  std::vector<std::tuple<int,int>> lines = hough_transform(img_data, w, h);
  */

  for (Vec2f elem: lines) {
    cout << elem << endl;	  
  }
}
