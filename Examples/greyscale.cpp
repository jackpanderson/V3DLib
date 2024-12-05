#include "V3DLib.h"
#include "Support/Settings.h"
#include <iostream>
#include <chrono>
#include <string.h>

using namespace V3DLib;
using namespace std;

// void greyscaleGPU(Int ops, Int::Ptr grey_o, Int::Ptr red, Int::Ptr green, Int::Ptr blue);

V3DLib::Settings settings;

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

int main(int argc, const char *argv[]) {
  //settings.init(argc, argv);
  initGPU();
  auto k = compile(greyscaleGPU);                       
  k.setNumQPUs(NUM_QPUS);

  int imgSize = 1280*590;

  char red[imgSize];
  char blue[imgSize];
  char green[imgSize];
  char grey[imgSize];


  for (int i = 0; i < imgSize; i+=1) {
    // we gotta fill the array char by char instead of int by int to make sure it works
    uint8_t rand_num = rand() % 256;
    red [i] = rand_num + (rand_num << 8) + (rand_num << 16) + (rand_num << 24);
    rand_num = rand() % 256;
    blue [i] = rand_num + (rand_num << 8) + (rand_num << 16) + (rand_num << 24);
    rand_num = rand() % 256;
    green [i] = rand_num + (rand_num << 8) + (rand_num << 16) + (rand_num << 24);
  }
  
  std::chrono::duration<double> elapsed_seconds;
  for (int i = 0; i < 10; i++) {
    // Invoke the kernel
    auto start = chrono::system_clock::now();
    doGPU(&k, red, green, blue, grey, imgSize >> 2);
    auto end = chrono::system_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed_ms = end-start;
    std::cout << "Duration (ms)" << (double)elapsed_ms.count()<< std::endl;

  }

  for (int i = 0; i < (int) 64; i++) {  // Display the result
  // for (int i = 0; i < (int) out_array.size(); i++) {  // Display the result
    //printf("%i: %X\n", i, grey[i]);
  }
    

  return 0;
}
