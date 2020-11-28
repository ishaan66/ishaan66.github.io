#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <tuple>
//#include <../opencv2-master/include/opencv2>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct hough_cmp_gt
{
    hough_cmp_gt(const int* _aux) : aux(_aux) {}
    inline bool operator()(int l1, int l2) const
    {
        return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
    }
    const int* aux;
};

void findLocalMaximums(int numrho, int numangle, int threshold, int *accum, std::vector<int> &sort_buf, unsigned int size) {
    for(int r = 0; r < numrho; r++ )
        for(int n = 0; n < numangle; n++ )
        {
            int base_index = r * (numangle) + n;
	    int left_index = (base_index - 1) < 0 ? -1 : base_index - 1;
	    int right_index = (base_index + 1) > size ? -1 : base_index - 1;
	    int up_index = (r > 0) ? (r-1) * (numangle) + n: -1;
	    int down_index = (r < numrho - 1) ? (r+1) * (numangle) + n: -1;
            if (accum[base_index] >= threshold &&
                (left_index == -1 || accum[base_index] > accum[base_index - 1]) && 
		(right_index == -1 || accum[base_index] >= accum[base_index + 1]) &&
                (up_index == -1 || accum[base_index] >= accum[up_index]) && 
		(down_index == -1 || accum[base_index] > accum[down_index]))
                sort_buf.push_back(base_index);
        }

}

std::vector<std::tuple<float, float>> hough_transform(Mat img_data, int w, int h) {
  // Create the accumulator
  int accum_height = 2 * (int) sqrt(w*w + h*h);
  int accum_width = 180;
  constexpr double PI = 3.14159;

  int *accum = new int[accum_height * accum_width];
  memset(accum, 0, sizeof(int) * accum_height * accum_width);

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
	//if (img_data[i*w + j] != 0) {
	if (img_data.at<float>(i,j) != 0.0f) {
	  for (int theta = 0; theta < accum_width; theta++) {
	    int rho = round(j * cos(theta * (PI / 180.0f)) + i * sin(theta * (PI / 180.0f)));
	    std::cout << "INIT RHO: " << rho;
	    rho += (accum_height - 1)/2;
	    std::cout << " INTERMEDIATE: " << (accum_height - 1) / 2 << " FINAL: " << rho << std::endl;
	    int index = (rho * accum_width) + theta;
	    //std::cout << "index is " << index << std::endl;
	    accum[index]++;
	  }
	}	
    }
  }

  std::vector<std::tuple<float, float>> lines;
  std::vector<int> sort_buf;
  // find local maximums
  findLocalMaximums(accum_height, accum_width, 2, accum, sort_buf, accum_width*accum_height);

  // stage 3. sort the detected lines by accumulator value
  std::sort(sort_buf.begin(), sort_buf.end(), hough_cmp_gt(accum));

  for (int index: sort_buf) {
    float rho = (index/accum_width) - ((accum_height - 1) / 2);
    float theta = (index % accum_width) * (PI / 180.0f);
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
  std::cout << "Image width: " << w << " Height: " << h << " Total: " << w*h << std::endl;
  delete accum;
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
    vector<Vec2f> cvlines; // will hold the results of the detection
    std::vector<std::tuple<float, float>> lines;
    HoughLines(dst, cvlines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection
    lines = hough_transform(dst, src.rows, src.cols);
    // Draw the lines
    for( size_t i = 0; i < min(lines.size(), cvlines.size()); i++ )
    {
        float cvrho = cvlines[i][0], cvtheta = cvlines[i][1];
	float rho = get<0>(lines[i]), theta = get<1>(lines[i]);	
	std::cout << "CV R: " << cvrho << " T: " << cvtheta << " MY R: " << rho << " T: " << theta << std::endl;
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line(cdst, pt1, pt2, Scalar(0,0,255), 3, 16);
    }

    imwrite("out_my_serial.jpg", cdst);
  return 0;
}


