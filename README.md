# gpt-in-cpp
A simple GPT style transformer model written in C++ using TorchLib

If you have libtorch installed in some path, can run 
```
g++ -std=c++14 train.cpp -o train -I /path/to/libtorch/include -L /path/to/libtorch/lib -ltorch -lc10
```

Then
```
./train
```
