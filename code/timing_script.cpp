#include <iostream>
#include <fstream>
#include <string>
#include "hough.cpp"
#include "418/hough_simd_basic.cpp"
#include "418/hough_simd.cpp"

double get_time_sec() {
  struct timeval curr_time;
  gettimeofday(&curr_time, NULL);
  return curr_time.tv_sec + curr_time.tv_usec / 1000000.0;
}

int main(int argc, char **argv) {
	if (argc != 3) {
		printf("incorrect arguments provided");
		return -1;
	}

	int flag = atoi(argv[1]);
	std::string outfile;
	if (flag == 0) {
		outfile = "execution_time_serial.csv";
	}
	if(flag == 1) {
		outfile = "execution_time_simd_basic.csv";
	}
	if (flag == 2) {
		outfile = "execution_time_simd.csv";
	}

	int num_iters = 50;

	Mat dst, cdst;
	Mat src = imread(argv[2], IMREAD_GRAYSCALE);
	int dy = 20;
	int dx = 20;
	int curr_rows = 20; //src.rows - 100*dy;
	int curr_cols =	20; //src.cols - 100*dx;

	std::cout << "curr rows: " << curr_rows << std::endl;
	std::cout << "curr cols: " << curr_cols << std::endl;

	// check if image is loaded fine
	if (src.empty()) {
		printf("Error opening image \n");
		return -1;
	}

	std::vector<std::tuple<float, float>> data;

	double t1;
	double t2;

	for (int i = 0; i < num_iters; i++) {
		std::cout << "iter: " << i << std::endl;
		if ((curr_rows < 20) || (curr_cols < 20)) {
			break;
		}
		Mat src_temp = src;
		resize(src_temp, src_temp, Size(curr_cols, curr_rows));

		Canny(src_temp, dst, 50, 200, 3);

		std::vector<std::tuple<float, float>> lines;
		
		if (flag == 0) {
			t1 = get_time_sec();
			lines = hough_transform_serial(dst, src_temp.cols, src_temp.rows, 100);
			t2 = get_time_sec();
		} else if (flag == 1) {
			t1 = get_time_sec();
			lines = hough_transform_simd_basic(dst, src_temp.cols, src_temp.rows, 100);
			t2 = get_time_sec();
		} else if (flag == 2) {
			t1 = get_time_sec();
			lines = hough_transform_simd(dst, src_temp.cols, src_temp.rows, 100);
			t2 = get_time_sec();
		}

		data.push_back(std::make_tuple(curr_rows * curr_cols, (t2 - t1)));

		curr_rows += 20;
		curr_cols += 20;
	}

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
