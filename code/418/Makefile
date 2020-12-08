all:
	g++ -std=c++11 -O3 -fopenmp -mavx2 -mfma -I/home/ishaan66/include/opencv -I/home/ishaan66/include/opencv2 -L/home/ishaan66/lib/ -g -o simd.o hough_simd.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

both:
	g++ -std=c++11 -O3 -fopenmp -mavx2 -mfma -I/home/ishaan66/include/opencv -I/home/ishaan66/include/opencv2 -L/home/ishaan66/lib/ -g -o openmp_simd.o hough_smp.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

openmp:
	g++ -std=c++11 -O3 -fopenmp -mavx2 -mfma -I/home/ishaan66/include/opencv -I/home/ishaan66/include/opencv2 -L/home/ishaan66/lib/ -g -o openmp.o hough_mp.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

simd:
	g++ -std=c++11 -O3 -fopenmp -mavx2 -mfma -I/home/ishaan66/include/opencv -I/home/ishaan66/include/opencv2 -L/home/ishaan66/lib/ -g -o simd.o hough_simd.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

simd_basic:
	g++ -std=c++11 -O3 -fopenmp -mavx2 -mfma -I/home/ishaan66/include/opencv -I/home/ishaan66/include/opencv2 -L/home/ishaan66/lib/ -g -o simd_basic.o hough_simd_basic.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

clean:
	rm -rf *.o *~
