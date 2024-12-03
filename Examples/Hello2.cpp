#include "V3DLib.h"
#include "Support/Settings.h"
#include <iostream>
#include <chrono>

using namespace V3DLib;
using namespace std;

V3DLib::Settings settings;

#define OPS 16

// VIDEOCODE KERNEL BABY
void hello3(Int ops, Int::Ptr grey_o, Int::Ptr red, Int::Ptr green, Int::Ptr blue) {                          // The kernel definition
  
  
  gather(red + me()*OPS);
  gather(green + me()*OPS);
  gather(blue + me()*OPS);
  

  Int red_int;
  Int green_int;
  Int blue_int;

  // BoolExpr::bexpr
  For ( Int i = me()*OPS , (i < ops*OPS) , i += OPS*numQPUs()) 
    // THIS IS A VECTOR OP ON 16 ELEMENTS RAAHHHHHHHHHHHHH
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
      //i forgor that the Int class is 32 bits always, and will grab 32 bits
      
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



int main(int argc, const char *argv[]) {
  settings.init(argc, argv);
  
  int size = 1280*256;
  int numQPUs = 8;

  auto k = compile(hello3);                        // Construct the kernel
  k.setNumQPUs(numQPUs);

  Int::Array grey(size);                           // Allocate and initialise the array shared between ARM and GPU
  Int::Array red(size);
  Int::Array green(size);
  Int::Array blue(size);
  
  // for (int i = 0; i < size; i++){
  //   red[i] = ((int*)rc)[i]
  // }

  for (int i = 0; i < size; i+=1) {
    // we gotta fill the array char by char instead of int by int to make sure it works
    int rand_num = rand() % 256;
    red [i] = rand_num + (rand_num << 8) + (rand_num << 16) + (rand_num << 24);
    rand_num = rand() % 256;
    blue [i] = rand_num + (rand_num << 8) + (rand_num << 16) + (rand_num << 24);
    rand_num = rand() % 256;
    green [i] = rand_num + (rand_num << 8) + (rand_num << 16) + (rand_num << 24);
    // yes this assigns the same number to the first 4 chars
  }
  
  int ops_per_qpu = size/(OPS);
  
  k.load(ops_per_qpu, &grey, &red, &green, &blue);
  
  std::chrono::duration<double> elapsed_seconds;
  for (int i = 0; i < 20; i++) {
  auto start = chrono::system_clock::now();
    // Invoke the kernel
  settings.process(k);  
  auto end = chrono::system_clock::now();
  elapsed_seconds = end-start;
  }

  for (int i = 0; i < (int) 32; i++) {  // Display the result
  // for (int i = 0; i < (int) out_array.size(); i++) {  // Display the result
    printf("%i: %X\n", i, grey[i]);
  }
  printf("elapsed time (ms): %.10f\n", elapsed_seconds.count() * 1000);

  return 0;
}
