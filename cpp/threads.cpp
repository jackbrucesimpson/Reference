#include <iostream>
#include <thread>

void f1() {
    std::cout << "This is a test of threads" << std::endl;
}

int main(){
    
    std::thread t1(f1); //t1 starts running
    //t1.join(); // main thread waits for t1 to finish

    // separate from main thread and runs on own: becomes daemon process
    // be careful: if main thread finishes first, may never have chance to finish
    t1.detach();
    // cannot join: its detached, but can check:
    if (t1.joinable()) {
        t1.join();
    }

  return 0;
}
