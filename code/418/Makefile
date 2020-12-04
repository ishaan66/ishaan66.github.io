all:
	g++ -O3 -std=c++11 -mavx2 -mfma -I/home/ranil/ishaan66@github.io/include/opencv -I/home/ranil/ishaan66@github.io/include/opencv2 -L/home/ranil/ishaan66@github.io/lib/ -g -o simd.o hough_simd.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

clean:
	rm -rf *.o *~
