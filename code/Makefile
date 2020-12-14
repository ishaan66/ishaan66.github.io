serial:
	g++ -std=c++11 -O3 -fopenmp -mavx2 -mfma -I/home/ranil/ishaan66.github.io/include/opencv -I/home/ranil/ishaan66.github.io/include/opencv2 -L/home/ranil/ishaan66.github.io/lib/ -g -o serial.o hough_serial.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

omp_simd:
	g++ -std=c++11 -O3 -fopenmp -mavx2 -mfma -I/home/ranil/ishaan66.github.io/include/opencv -I/home/ranil/ishaan66.github.io/include/opencv2 -L/home/ranil/ishaan66.github.io/lib/ -g -o omp_simd.o hough_omp_simd.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

simd_basic:
	g++ -std=c++11 -O3 -fopenmp -mavx2 -mfma -I/home/ranil/ishaan66.github.io/include/opencv -I/home/ranil/ishaan66.github.io/include/opencv2 -L/home/ranil/ishaan66.github.io/lib/ -g -o simd_basic.o hough_simd_basic.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

simd_adv:
	g++ -std=c++11 -O3 -fopenmp -mavx2 -mfma -I/home/ranil/ishaan66.github.io/include/opencv -I/home/ranil/ishaan66.github.io/include/opencv2 -L/home/ranil/ishaan66.github.io/lib/ -g -o simd_adv.o hough_simd_adv.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

omp:
	g++ -std=c++11 -O3 -fopenmp -mavx2 -mfma -I/home/ranil/ishaan66.github.io/include/opencv -I/home/ranil/ishaan66.github.io/include/opencv2 -L/home/ranil/ishaan66.github.io/lib/ -g -o omp.o hough_omp.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

clean:
	rm -rf *.o *~
