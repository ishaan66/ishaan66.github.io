all:
	g++ -std=c++11 -O3 -mavx2 -mfma -I/home/ishaan66/include/opencv -I/home/ishaan66/include/opencv2 -L/home/ishaan66/lib/ -g -o simd.o hough_simd.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

clean:
	rm -rf *.o *~
