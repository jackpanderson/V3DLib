#include "V3DLib.h"
#include "Support/Settings.h"
#include <iostream>
#include <chrono>
#include <string.h>

#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <pthread.h>
#include <arm_neon.h>
#include <chrono>

using namespace cv;

int16x8_t convolveSobelX(Mat img, int row, int col);
int16x8_t convolveSobelY(Mat img, int row, int col);
int16x8_t vectorByScalar(int row, int col, int16_t scalar);
void * sobelThread(void * args);

using namespace V3DLib;
using namespace std;


V3DLib::Settings settings;
unsigned int nthreads;
pthread_barrier_t doneBarrier, newBarrier;
Mat img, sobelX, sobelY, sobelOut;
volatile int exitFlag = 0;

#define OPS 16
#define NUM_QPUS 8

// VIDEOCODE KERNEL BABY
void greyscaleGPU(Int ops, Int::Ptr grey_o, Int::Ptr red, Int::Ptr green, Int::Ptr blue) {                          // The kernel definition
  
  
  gather(red + me()*OPS);
  gather(green + me()*OPS);
  gather(blue + me()*OPS);
  

  Int red_int;
  Int green_int;
  Int blue_int;

  // BoolExpr::bexpr
  For ( Int i = me()*OPS , (i < ops*OPS) , i += OPS*numQPUs()) 
    // Load Ints from Memory
    receive(red_int);
    receive(green_int);
    receive(blue_int);

    gather(red + i + OPS*numQPUs());
    gather(green + i + OPS*numQPUs());
    gather(blue + i + OPS*numQPUs());

    auto grey_int = Int(0);
    // Break Int Into 4 Chars
    for (int j = 0; j < 32; j+=8)
    {
      // Get appropriate char for each color
      auto red_char    = (Int(red_int)    >> (j)) & 0xFF;
      auto green_char  = (Int(green_int)  >> (j)) & 0xFF;
      auto blue_char   = (Int(blue_int)   >> (j)) & 0xFF;
      // Greyscale Conversion on Char
      auto grey_char  = toInt(toFloat(red_char) * 0.299)
                      + toInt(toFloat(green_char) * 0.587)
                      + toInt(toFloat(blue_char) * 0.144);
      // Put char back into its spot in the int
      
      grey_int |= (grey_char << (j));
    }

    grey_o[i] = grey_int;

  End

  receive(red_int);
  receive(green_int);
  receive(blue_int);

}

// VIDEOCODE KERNEL BABY
void hello2(Int ops, Int::Ptr grey_o, Int::Ptr red, Int::Ptr green, Int::Ptr blue) {                          // The kernel definition
  // BoolExpr::bexpr
  For ( Int i = me()*OPS , (i < ops*OPS) , i += OPS*numQPUs()) 
    // THIS IS A VECTOR OP ON 16 ELEMENTS RAAHHHHHHHHHHHHH
    // Load Ints from Memory
    auto red_int = red[i];
    auto green_int = green[i];
    auto blue_int = blue[i];
    auto grey_int = Int(0);
    // Break Int Into 4 Chars
    for (int j = 0; j < 32; j+=8)
    {
      // Get appropriate char for each color
      auto red_char    = (Int(red_int)    >> (j)) & 0xFF;
      auto green_char  = (Int(green_int)  >> (j)) & 0xFF;
      auto blue_char   = (Int(blue_int)   >> (j)) & 0xFF;
      // Greyscale Conversion on Char
      auto grey_char  = toInt(toFloat(red_char) * 0.299)
                      + toInt(toFloat(green_char) * 0.587)
                      + toInt(toFloat(blue_char) * 0.144);
      // Put char back into its spot in the int
      //i forgor that the Int class is 32 bits always, and will grab 32 bits
      
      grey_int |= (grey_char << (j));
    }
    
    grey_o[i] = grey_int;

  End
}

// VIDEOCODE KERNEL BABY
void hello1(Int ops, Int::Ptr grey_o, Int::Ptr red, Int::Ptr green, Int::Ptr blue) {                          // The kernel definition
  // BoolExpr::bexpr
  For ( Int i = me()*OPS , (i < ops*OPS) , i += OPS*numQPUs()) 
    // THIS IS A VECTOR OP ON 16 ELEMENTS RAAHHHHHHHHHH
    grey_o[i] = toInt(toFloat(Int(red[i])) * 0.299)
              + toInt(toFloat(Int(green[i])) * 0.587)
              + toInt(toFloat(Int(blue[i])) * 0.144);
  End
}

//template <typename... ts>
void initGPU () {
  const char* arr[] = {"idk"};
  settings.init(1, arr);
}

template <typename... ts>
void doGPU(Kernel<ts...> * k, char * red, char * green, char * blue, char * grey, int vectorSize) { //vector size should be (total image pixels)/4
  std::chrono::duration<double> elapsed_seconds;
  
  Int::Array vGrey(vectorSize); 
  Int::Array vRed(vectorSize);
  Int::Array vGreen(vectorSize);
  Int::Array vBlue(vectorSize);

  memcpy((int32_t *)(vRed.ptr()), (int32_t *) red, vectorSize * sizeof(int32_t));
  memcpy((int32_t *)(vGreen.ptr()), (int32_t *) green, vectorSize * sizeof(int32_t));
  memcpy((int32_t *)(vBlue.ptr()), (int32_t *) blue, vectorSize * sizeof(int32_t));

  int ops_per_qpu = (vectorSize << 2)/(OPS);
  
  (*k).load(ops_per_qpu, &vGrey, &vRed, &vGreen, &vBlue);
  auto start = chrono::system_clock::now();
  settings.process(*k);
  auto end = chrono::system_clock::now();
  elapsed_seconds = end-start;
  printf("Kernel Process elapsed time (ms): %.10f\n", elapsed_seconds.count() * 1000);

  memcpy(grey, (int32_t *)(vGrey.ptr()), vectorSize * sizeof(int32_t));
}

// int main(int argc, const char *argv[]) {
//   //settings.init(argc, argv);
//   initGPU();
//   auto k = compile(greyscaleGPU);                       
//   k.setNumQPUs(NUM_QPUS);
//   int imgSize = 1280*590;
//   char red[imgSize];
//   char blue[imgSize];
//   char green[imgSize];
//   char grey[imgSize];


//   for (int i = 0; i < imgSize; i+=1) {
//     // we gotta fill the array char by char instead of int by int to make sure it works
//     uint8_t rand_num = rand() % 256;
//     red [i] = rand_num + (rand_num << 8) + (rand_num << 16) + (rand_num << 24);
//     rand_num = rand() % 256;
//     blue [i] = rand_num + (rand_num << 8) + (rand_num << 16) + (rand_num << 24);
//     rand_num = rand() % 256;
//     green [i] = rand_num + (rand_num << 8) + (rand_num << 16) + (rand_num << 24);
//   }
  
//   std::chrono::duration<double> elapsed_seconds;
//   for (int i = 0; i < 10; i++) {
//     // Invoke the kernel
//     auto start = chrono::system_clock::now();
//     doGPU(&k, red, green, blue, grey, imgSize >> 2);
//     auto end = chrono::system_clock::now();
    
//     std::chrono::duration<double, std::milli> elapsed_ms = end-start;
//     std::cout << "Duration (ms)" << (double)elapsed_ms.count()<< std::endl;

//   }

//   for (int i = 0; i < (int) 64; i++) {  // Display the result
//   // for (int i = 0; i < (int) out_array.size(); i++) {  // Display the result
//     //printf("%i: %X\n", i, grey[i]);
//   }
    

//   return 0;
// }

void * sobelThread(void * args)
{
  int64_t thdNum = (int64_t) args;

  int blockSize = (img.rows + nthreads - 1) / (nthreads);
  int rowStart = thdNum * blockSize;
  rowStart = (rowStart == 0) ? 1 : rowStart;
  int rowStop = rowStart + blockSize;
  rowStop = (img.rows-1 < rowStop) ? img.rows-1 : rowStop;
  
  while (exitFlag == 0) {
    pthread_barrier_wait(&newBarrier);
    int row, col;
    // Calculate sobel x and sobel y
    for (row = rowStart; row < rowStop; row++) {
        for (int col = 1; col < img.cols-1; col += 8) {
          // Sobel X and Y
          int16x8_t sumVecX = convolveSobelX(img, row, col);
          int16x8_t sumVecY = convolveSobelY(img, row, col);
          // Magnitude
          sumVecX = vabsq_s16(sumVecX);
          sumVecY = vabsq_s16(sumVecY);
          int16x8_t mag = vaddq_s16(sumVecX, sumVecY);
          // Truncate s16 to u8, clamping to 255
          uint8x8_t output = vqmovun_s16(mag);
          uint8_t * outputPtr = sobelOut.ptr<uint8_t>(row, col);
          vst1_u8(outputPtr, output);
          // mag = (mag > 255) ? 255 : mag;
          // sobelOut.at<uint8_t>(row, col) = (uint8_t) mag;
        }
    }
    pthread_barrier_wait(&doneBarrier);

  }
}


int16x8_t convolveSobelX(Mat img, int row, int col) {
  uint8x8_t gray8Vec;
  int16x8_t gray16Vec;
  int16x8_t sumVec = {0};

  sumVec = vaddq_s16(sumVec, vectorByScalar(row + 1, col - 1,  1));
  sumVec = vaddq_s16(sumVec, vectorByScalar(row + 0, col - 1,  2));
  sumVec = vaddq_s16(sumVec, vectorByScalar(row - 1, col - 1,  1));
  sumVec = vaddq_s16(sumVec, vectorByScalar(row + 1, col + 1, -1));
  sumVec = vaddq_s16(sumVec, vectorByScalar(row + 0, col + 1, -2));
  sumVec = vaddq_s16(sumVec, vectorByScalar(row - 1, col - 1, -1));

  return sumVec;
};

int16x8_t convolveSobelY(Mat img, int row, int col) {
  uint8x8_t gray8Vec;
  int16x8_t gray16Vec;
  int16x8_t sumVec = {0};

  sumVec = vaddq_s16(sumVec, vectorByScalar(row - 1, col - 1,  1));
  sumVec = vaddq_s16(sumVec, vectorByScalar(row - 1, col + 0,  2));
  sumVec = vaddq_s16(sumVec, vectorByScalar(row - 1, col + 1,  1));
  sumVec = vaddq_s16(sumVec, vectorByScalar(row + 1, col - 1, -1));
  sumVec = vaddq_s16(sumVec, vectorByScalar(row + 1, col + 0, -2));
  sumVec = vaddq_s16(sumVec, vectorByScalar(row + 1, col - 1, -1));

  return sumVec;

};

int16x8_t vectorByScalar(int row, int col, int16_t scalar) {
  uint8x8_t gray8Vec;
  int16x8_t gray16Vec;

  // Load 8x8 vector from current location
  gray8Vec = vld1_u8(img.ptr<uint8_t>(row, col));
  
  // Shift 8x8 vector into 8x16 vector
  gray16Vec = vreinterpretq_s16_u16(vmovl_u8(gray8Vec));
  if (scalar != 1) {
    // Multiply s16 vector by constant
    gray16Vec = vmulq_n_s16(gray16Vec, scalar);
  }
  return gray16Vec;
}

int main(int argc, char ** argv)
{
  initGPU();
  auto k = compile(greyscaleGPU);                       
  k.setNumQPUs(NUM_QPUS);

  nthreads = std::thread::hardware_concurrency();
  
  char * vidName;
  if (argc == 1) {
    vidName = "mikeU.avi";
  } else {
    vidName = argv[1];
  }

  VideoCapture mikeVid(vidName);

  if (!mikeVid.isOpened())
  {
    std::cout << "Video file failed to open..." << std::endl;
    return EXIT_FAILURE;
  }
  
  Mat frame;
  mikeVid >> frame;
  Mat colorChannels[3];
  split(frame, colorChannels);

  auto start = std::chrono::system_clock::now();
    // CCIR 601 Grayscale Conversion RGB To Luma
  img = colorChannels[2] * 0.299 
                  + colorChannels[1] * 0.587 
                  + colorChannels[0] * 0.144;

  Mat img1 = img.clone();
  Mat img2 = img.clone();
  int count = 0;

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "OpenCV Duration (ms)" << elapsed_seconds.count() * 1000 << std::endl;


  sobelOut    = Mat::zeros(Size(frame.cols,frame.rows),CV_8UC1);

  int doneBarr = pthread_barrier_init(&doneBarrier, NULL, nthreads + 1);
  int newBarr = pthread_barrier_init(&newBarrier, NULL, nthreads + 1);

  pthread_t threads[nthreads];
  for (int thdNum = 0; thdNum < nthreads; thdNum++) {
    pthread_create(&threads[thdNum],
                          NULL,
                          sobelThread,
                          (void *)thdNum);
  }

  while(1)
  {
    auto start = std::chrono::system_clock::now();

    if (frame.empty()) {
      pthread_barrier_wait(&newBarrier);
      exitFlag = 1;
      pthread_barrier_wait(&doneBarrier);
      break;
    }

    // split(frame, colorChannels);
    // // Mat gpuIMG = (count % 2 == 1) ? img2 : img1;
    // // CCIR 601 Grayscale Conversion RGB To Luma
    // doGPU(&k, colorChannels[0].ptr<char>(0,0), colorChannels[1].ptr<char>(0,0), colorChannels[2].ptr<char>(0,0)
    // , img.ptr<char>(0,0), (frame.cols*frame.rows) >> 2);
    
    // Main gets here and lets loose all the workers
    pthread_barrier_wait(&newBarrier);

    split(frame, colorChannels);
    Mat gpuIMG = (count % 2 == 1) ? img2 : img1;
    // CCIR 601 Grayscale Conversion RGB To Luma
    doGPU(&k, colorChannels[0].ptr<char>(0,0), colorChannels[1].ptr<char>(0,0), colorChannels[2].ptr<char>(0,0)
    , gpuIMG.ptr<char>(0,0), (frame.cols*frame.rows) >> 2);

      
    // All worker threads are working here
    pthread_barrier_wait(&doneBarrier);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end-start;
    std::cout << "Total Loop Duration (ms)" << (double)elapsed_ms.count()<< std::endl;
    // All the workers are now waiting for new frame


    imshow("FilteredVideo", sobelOut); //show filtered frame

    if (waitKey(1) == 'q') {//approx 24fps, 42ms/frame
      pthread_barrier_wait(&newBarrier);
      exitFlag = 1;
      pthread_barrier_wait(&doneBarrier);
      break;
    }

    mikeVid >> frame;
    count++;
    img = (count % 2 == 0) ? img2 : img1;
  }
  void * ret;
  for (int thdNum = 0; thdNum < nthreads; thdNum++) {
    pthread_join(threads[thdNum], &ret);
  }

  mikeVid.release(); //call destructor
  destroyAllWindows();
  return 0; 
}

