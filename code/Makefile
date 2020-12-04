all:
	g++ -std=c++11 -O3 -mavx2 -mfma -I/home/ishaan66/include/opencv -I/home/ishaan66/include/opencv2 -L/home/ishaan66/lib/ -g -o my_serial.o hough.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

clean:
	rm -rf *.o *~
