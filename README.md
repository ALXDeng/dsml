# Distributed Machine Learning

# Running the Project 

**Installing necessary dependencies:**

go mod tidy

**Starting Services:**

go run cmd/device/main.go --port 50051

go run cmd/coordinator/main.go --port 50052

**Testing:**

go test ./pkg/tests -v

**For multiple runs of our testing script:**

./test.sh


# Group Work
As a suite, we all worked concurrently on this project on one laptop. We worked together to implement the RPC calls and building out the services. Then, we moved on to working on the AllReduceRing and NaiveAllReduce algorithms. Throughout the lab, we worked together to implement unit testing to ensure that our implementation was working as intended, finishing with a thorough time complexity comparison test between our two algorithms. Finally we completed the write-up.


# Sample Output of Our Implementation + Unit Tests

```
JZhengs-MacBook-Pro:dsml jasonzheng$ go test ./pkg/tests -race -v
=== RUN   TestInitCommunicator
--- PASS: TestInitCommunicator (0.09s)
=== RUN   TestMemcpyToGPUs
--- PASS: TestMemcpyToGPUs (0.08s)
=== RUN   TestGroupOperations
--- PASS: TestGroupOperations (0.08s)
=== RUN   TestAllReduce
--- PASS: TestAllReduce (0.19s)
=== RUN   TestStatusChecking
--- PASS: TestStatusChecking (0.09s)
=== RUN   TestMemcpyFromGPU
--- PASS: TestMemcpyFromGPU (0.09s)
=== RUN   TestAllReduceRing
--- PASS: TestAllReduceRing (0.19s)
=== RUN   TestNaiveAllReduce
--- PASS: TestNaiveAllReduce (0.09s)
=== RUN   TestAllReducePerformance
=== RUN   TestAllReducePerformance/Tiny
    gpu_test.go:1183: 
        Performance comparison for Tiny size (1000 elements, 0.01 MB), averaged over 5 iterations:
    gpu_test.go:1185: Naive AllReduce: 7.418316ms
    gpu_test.go:1186: Ring AllReduce:  8.172442ms
    gpu_test.go:1187: Speedup: 0.91x
=== RUN   TestAllReducePerformance/Small
    gpu_test.go:1183: 
        Performance comparison for Small size (10000 elements, 0.08 MB), averaged over 5 iterations:
    gpu_test.go:1185: Naive AllReduce: 26.070249ms
    gpu_test.go:1186: Ring AllReduce:  12.05425ms
    gpu_test.go:1187: Speedup: 2.16x
=== RUN   TestAllReducePerformance/Medium
    gpu_test.go:1183: 
        Performance comparison for Medium size (50000 elements, 0.38 MB), averaged over 3 iterations:
    gpu_test.go:1185: Naive AllReduce: 97.468347ms
    gpu_test.go:1186: Ring AllReduce:  31.570236ms
    gpu_test.go:1187: Speedup: 3.09x
=== RUN   TestAllReducePerformance/Large
    gpu_test.go:1183: 
        Performance comparison for Large size (100000 elements, 0.76 MB), averaged over 2 iterations:
    gpu_test.go:1185: Naive AllReduce: 196.573604ms
    gpu_test.go:1186: Ring AllReduce:  55.201979ms
    gpu_test.go:1187: Speedup: 3.56x
=== RUN   TestAllReducePerformance/XLarge
    gpu_test.go:1183: 
        Performance comparison for XLarge size (200000 elements, 1.53 MB), averaged over 2 iterations:
    gpu_test.go:1185: Naive AllReduce: 377.395916ms
    gpu_test.go:1186: Ring AllReduce:  106.632396ms
    gpu_test.go:1187: Speedup: 3.54x
=== RUN   TestAllReducePerformance/XXLarge
    gpu_test.go:1183: 
        Performance comparison for XXLarge size (450000 elements, 3.43 MB), averaged over 1 iterations:
    gpu_test.go:1185: Naive AllReduce: 854.384375ms
    gpu_test.go:1186: Ring AllReduce:  224.469292ms
    gpu_test.go:1187: Speedup: 3.81x
--- PASS: TestAllReducePerformance (7.71s)
    --- PASS: TestAllReducePerformance/Tiny (1.01s)
    --- PASS: TestAllReducePerformance/Small (1.17s)
    --- PASS: TestAllReducePerformance/Medium (1.08s)
    --- PASS: TestAllReducePerformance/Large (1.06s)
    --- PASS: TestAllReducePerformance/XLarge (1.71s)
    --- PASS: TestAllReducePerformance/XXLarge (1.67s)
PASS
ok      dsml/pkg/tests  9.914s
JZhengs-MacBook-Pro:dsml jasonzheng$ 
```