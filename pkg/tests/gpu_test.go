// pkg/tests/gpu_test.go
package tests

import (
	"context"
	"encoding/binary"
	"log"
	"math"
	"net"
	"testing"
	"time"

	"dsml/pkg/coordinator"
	pb "dsml/proto/gpu_sim"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)


func setupTest(t *testing.T) (pb.GPUCoordinatorClient, *coordinator.GPUCoordinator, func()) {
    //start the server
    serverAddr, gpuCoordinator, serverCleanup := startServer(t)

    var opts []grpc.DialOption
    opts = append(opts, 
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithDefaultCallOptions(
            grpc.MaxCallRecvMsgSize(1024*1024*1024), 
            grpc.MaxCallSendMsgSize(1024*1024*1024), 
        ),
    )

    // Create client connection
    conn, err := grpc.Dial(serverAddr, opts...)
    if err != nil {
        time.Sleep(10 * time.Millisecond)
        conn, err = grpc.Dial(serverAddr, opts...)
        if err != nil {
            t.Fatalf("Failed to create client connection: %v", err)
        }
    }

    client := pb.NewGPUCoordinatorClient(conn)
    
    cleanup := func() {
        conn.Close()
        serverCleanup()
    }

    return client, gpuCoordinator, cleanup
}

func startServer(t *testing.T) (string, *coordinator.GPUCoordinator, func()) {
    listener, err := net.Listen("tcp", ":0")
    if err != nil {
        t.Fatalf("Failed to listen: %v", err)
    }
    
    grpcServer := grpc.NewServer(
        grpc.MaxRecvMsgSize(1024*1024*1024), 
        grpc.MaxSendMsgSize(1024*1024*1024), 
    )
    
    coord := coordinator.NewGPUCoordinator()
    pb.RegisterGPUCoordinatorServer(grpcServer, coord)

    go func() {
        if err := grpcServer.Serve(listener); err != nil {
            log.Printf("Server exited with error: %v", err)
        }
    }()

    addr := listener.Addr().String()
    cleanup := func() {
        grpcServer.GracefulStop()
        listener.Close()
    }

    return addr, coord, cleanup
}


// Test 1: Initialize Communicator
func TestInitCommunicator(t *testing.T) {
    client, _, cleanup := setupTest(t)
    defer cleanup()
    
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    N := 4 // number of GPUs
    resp, err := client.CommInit(ctx, &pb.CommInitRequest{
        NumDevices: uint32(N),
    })

    if err != nil {
        t.Fatalf("CommInit failed: %v", err)
    }
    if !resp.Success {
        t.Fatal("CommInit reported failure")
    }
    if len(resp.Devices) != N {
        t.Errorf("Expected %d devices, got %d", N, len(resp.Devices))
    }
}

func TestMemcpyToGPUs(t *testing.T) {
    client, gpuCoordinator, cleanup := setupTest(t)
    defer cleanup()

    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    //initialize communicator
    N := 4
    commInit, err := client.CommInit(ctx, &pb.CommInitRequest{
        NumDevices: uint32(N),
    })
    if err != nil {
        t.Fatalf("Failed to initialize communicator: %v", err)
    }

    //create test data
    testData := make([]float64, 1024)
    for i := range testData {
        testData[i] = float64(i)
    }

    //convert to bytes
    dataBytes := make([]byte, len(testData)*8)
    for i, v := range testData {
        binary.LittleEndian.PutUint64(dataBytes[i*8:], math.Float64bits(v))
    }

    //test copying to each GPU
    for i := 0; i < N; i++ {
        resp, err := client.Memcpy(ctx, &pb.MemcpyRequest{
            Either: &pb.MemcpyRequest_HostToDevice{
                HostToDevice: &pb.MemcpyHostToDeviceRequest{
                    HostSrcData: dataBytes,
                    DstDeviceId: commInit.Devices[i].DeviceId,
                    DstMemAddr:  commInit.Devices[i].MinMemAddr,
                },
            },
        })
        if err != nil {
            t.Errorf("Failed to copy data to GPU %d: %v", i, err)
            continue
        }
        if !resp.GetHostToDevice().Success {
            t.Errorf("Memcpy to GPU %d reported failure", i)
            continue
        }

        //validate data on GPU

        deviceID := commInit.Devices[i].DeviceId.Value
        gpuDevice, exists := gpuCoordinator.Devices[deviceID]
        if !exists {
            t.Errorf("Device with ID %d not found for validation", deviceID)
            continue
        }

        for j := 0; j < len(testData); j++ {
            addr := commInit.Devices[i].MinMemAddr.Value + uint64(j*8)
            actual := math.Float64frombits(binary.LittleEndian.Uint64(gpuDevice.Memory[addr : addr+8]))
            
            if actual != testData[j] {
                t.Errorf("Data mismatch on GPU %d at index %d: got %f, expected %f", i, j, actual, testData[j])
                break
            }
        }
    }
}


// Test 3: Group Operations
func TestGroupOperations(t *testing.T) {
    client, _, cleanup := setupTest(t)
    defer cleanup()
    ctx := context.Background()

    //initialize communicator
    commInit, err := client.CommInit(ctx, &pb.CommInitRequest{
        NumDevices: 4,
    })
    if err != nil {
        t.Fatalf("Failed to initialize communicator: %v", err)
    }

    //test GroupStart
    startResp, err := client.GroupStart(ctx, &pb.GroupStartRequest{
        CommId: commInit.CommId,
    })
    if err != nil {
        t.Fatalf("GroupStart failed: %v", err)
    }
    if !startResp.Success {
        t.Error("GroupStart reported failure")
    }

    //test GroupEnd
    endResp, err := client.GroupEnd(ctx, &pb.GroupEndRequest{
        CommId: commInit.CommId,
    })
    if err != nil {
        t.Fatalf("GroupEnd failed: %v", err)
    }
    if !endResp.Success {
        t.Error("GroupEnd reported failure")
    }
}

// Test 4: AllReduce Operation
func TestAllReduce(t *testing.T) {
    client, _, cleanup := setupTest(t)
    defer cleanup()
    
    //use a longer timeout for this test
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    //initialize communicator
    N := 4
    commInit, err := client.CommInit(ctx, &pb.CommInitRequest{
        NumDevices: uint32(N),
    })
    if err != nil {
        t.Fatalf("Failed to initialize communicator: %v", err)
    }

    //give some time for device connections to stabilize
    time.Sleep(100 * time.Millisecond)
    //prepare memory addresses for AllReduce
    memAddrs := make(map[uint32]*pb.MemAddr)
    for i := uint32(0); i < uint32(N); i++ {
        memAddrs[i] = commInit.Devices[i].MinMemAddr
    }

    //start group operation
    _, err = client.GroupStart(ctx, &pb.GroupStartRequest{
        CommId: commInit.CommId,
    })
    if err != nil {
        t.Fatalf("GroupStart failed: %v", err)
    }

    //perform AllReduce
    allReduceResp, err := client.AllReduceRing(ctx, &pb.AllReduceRingRequest{
        CommId:   commInit.CommId,
        Count:    1024 * 8, // 1024 float64 values
        Op:       pb.ReduceOp_SUM,
        MemAddrs: memAddrs,
    })
    if err != nil {
        t.Fatalf("AllReduce failed: %v", err)
    }
    if !allReduceResp.Success {
        t.Error("AllReduce reported failure")
    }

    //end group operation
    _, err = client.GroupEnd(ctx, &pb.GroupEndRequest{
        CommId: commInit.CommId,
    })
    if err != nil {
        t.Fatalf("GroupEnd failed: %v", err)
    }
}

// Test 5: Status Checking
func TestStatusChecking(t *testing.T) {
    client, _, cleanup := setupTest(t)
    defer cleanup()
    ctx := context.Background()

    //initialize communicator
    commInit, err := client.CommInit(ctx, &pb.CommInitRequest{
        NumDevices: 4,
    })
    if err != nil {
        t.Fatalf("Failed to initialize communicator: %v", err)
    }

    //check initial status
    status, err := client.GetCommStatus(ctx, &pb.GetCommStatusRequest{
        CommId: commInit.CommId,
    })
    if err != nil {
        t.Fatalf("Failed to get status: %v", err)
    }
    if status.Status != pb.Status_IN_PROGRESS {
        t.Errorf("Expected initial status IN_PROGRESS, got %v", status.Status)
    }
}

// Test 6: Memory Copy from GPU back to CPU
 func TestMemcpyFromGPU(t *testing.T) {
    client, _, cleanup := setupTest(t)
    defer cleanup()

    ctx := context.Background()

    //initialize communicator
    commInit, err := client.CommInit(ctx, &pb.CommInitRequest{
        NumDevices: 4,
    })
    if err != nil {
        t.Fatalf("Failed to initialize communicator: %v", err)
    }

    //create test data and copy to GPU
    testData := make([]float64, 1024)
    for i := range testData {
        testData[i] = float64(i)
    }

    //convert test data to bytes
    dataBytes := make([]byte, len(testData)*8)
    for i, v := range testData {
        binary.LittleEndian.PutUint64(dataBytes[i*8:], math.Float64bits(v))
    }

    _, err = client.Memcpy(ctx, &pb.MemcpyRequest{
        Either: &pb.MemcpyRequest_HostToDevice{
            HostToDevice: &pb.MemcpyHostToDeviceRequest{
                HostSrcData: dataBytes,
                DstDeviceId: commInit.Devices[0].DeviceId,
                DstMemAddr:  commInit.Devices[0].MinMemAddr,
            },
        },
    })
    if err != nil {
        t.Fatalf("Failed to copy data to GPU: %v", err)
    }

    //copy data back from GPU
    resp, err := client.Memcpy(ctx, &pb.MemcpyRequest{
        Either: &pb.MemcpyRequest_DeviceToHost{
            DeviceToHost: &pb.MemcpyDeviceToHostRequest{
                SrcDeviceId: commInit.Devices[0].DeviceId,
                SrcMemAddr:  commInit.Devices[0].MinMemAddr,
                NumBytes:    uint64(len(dataBytes)),
            },
        },
    })
    if err != nil {
        t.Fatalf("Failed to copy data from GPU: %v", err)
    }

    copiedData := resp.GetDeviceToHost().DstData
    for i := 0; i < len(testData); i++ {
        actual := math.Float64frombits(binary.LittleEndian.Uint64(copiedData[i*8:]))
        // t.Logf("Data copied at index %d: got %f, expected %f", i, actual, testData[i])
        if actual != testData[i] {
            t.Errorf("Data mismatch at index %d: got %f, expected %f", i, actual, testData[i])
        }
    }
}

func TestCompleteWorkflow(t *testing.T) {
    client, _, cleanup := setupTest(t)
    defer cleanup()

    ctx := context.Background()

    //setup parameters
    N := 4          
    vectorSize := 8 
    bytesPerFloat := 8
    totalBytes := uint64(vectorSize * bytesPerFloat)

    //initialize communicator
    commInitResp, err := client.CommInit(ctx, &pb.CommInitRequest{NumDevices: uint32(N)})
    if err != nil {
        t.Fatalf("CommInit failed: %v", err)
    }

    // Create and initialize vectors on each GPU
    for deviceRank := 0; deviceRank < N; deviceRank++ {
        // Create vector where all elements equal rank+1
        data := make([]byte, totalBytes)
        value := float64(deviceRank + 1)
        
        // Fill entire vector with this value
        for i := 0; i < vectorSize; i++ {
            binary.LittleEndian.PutUint64(data[i*8:], math.Float64bits(value))
        }

        //copy to GPU
        _, err := client.Memcpy(ctx, &pb.MemcpyRequest{
            Either: &pb.MemcpyRequest_HostToDevice{
                HostToDevice: &pb.MemcpyHostToDeviceRequest{
                    HostSrcData: data,
                    DstDeviceId: commInitResp.Devices[deviceRank].DeviceId,
                    DstMemAddr:  commInitResp.Devices[deviceRank].MinMemAddr,
                },
            },
        })
        if err != nil {
            t.Fatalf("Failed to copy data to GPU %d: %v", deviceRank, err)
        }
    }

    //initialize memory addresses map for AllReduce
    memAddrs := make(map[uint32]*pb.MemAddr)
    for i := uint32(0); i < uint32(N); i++ {
        memAddrs[i] = commInitResp.Devices[i].MinMemAddr
    }

    //start group operation
    if _, err := client.GroupStart(ctx, &pb.GroupStartRequest{CommId: commInitResp.CommId}); err != nil {
        t.Fatalf("GroupStart failed: %v", err)
    }

    // Execute AllReduce
    allReduceResp, err := client.AllReduceRing(ctx, &pb.AllReduceRingRequest{
        CommId:   commInitResp.CommId,
        Count:    totalBytes,
        Op:       pb.ReduceOp_SUM,
        MemAddrs: memAddrs,
    })
    if err != nil {
        t.Fatalf("AllReduce failed: %v", err)
    }
    if !allReduceResp.Success {
        t.Fatal("AllReduce reported failure")
    }

    //end group operation
    if _, err := client.GroupEnd(ctx, &pb.GroupEndRequest{CommId: commInitResp.CommId}); err != nil {
        t.Fatalf("GroupEnd failed: %v", err)
    }

    time.Sleep(100 * time.Millisecond)

    //copy result back from GPU 0 and verify
    resp, err := client.Memcpy(ctx, &pb.MemcpyRequest{
        Either: &pb.MemcpyRequest_DeviceToHost{
            DeviceToHost: &pb.MemcpyDeviceToHostRequest{
                SrcDeviceId: commInitResp.Devices[0].DeviceId,
                SrcMemAddr:  commInitResp.Devices[0].MinMemAddr,
                NumBytes:    totalBytes,
            },
        },
    })
    if err != nil {
        t.Fatalf("Failed to copy result from GPU: %v", err)
    }

    //convert result back to float64 slice and verify
    result := make([]float64, vectorSize)
    data := resp.GetDeviceToHost().DstData
    
    //print the raw bytes for debugging
    
    
    for i := range result {
        result[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
        // Expected sum should be 1 + 2 + 3 + 4 = 10
        if math.Abs(result[i]-10.0) > 1e-10 {
            t.Errorf("Result[%d] = %f, want 10.0", i, result[i])
        }
    }
}

func TestNaiveAllReduce(t *testing.T) {
    client, _, cleanup := setupTest(t)
    defer cleanup()

    ctx := context.Background()

    //setup parameters
    N := 4         
    vectorSize := 8 
    bytesPerFloat := 8
    totalBytes := uint64(vectorSize * bytesPerFloat)

    //initialize communicator
    commInitResp, err := client.CommInit(ctx, &pb.CommInitRequest{NumDevices: uint32(N)})
    if err != nil {
        t.Fatalf("CommInit failed: %v", err)
    }

    //initialize each device with its rank+1 as values
    for deviceRank := 0; deviceRank < N; deviceRank++ {
        data := make([]byte, totalBytes)
        value := float64(deviceRank + 1)
        
        for i := 0; i < vectorSize; i++ {
            binary.LittleEndian.PutUint64(data[i*8:], math.Float64bits(value))
        }

        
        
        _, err := client.Memcpy(ctx, &pb.MemcpyRequest{
            Either: &pb.MemcpyRequest_HostToDevice{
                HostToDevice: &pb.MemcpyHostToDeviceRequest{
                    HostSrcData: data,
                    DstDeviceId: commInitResp.Devices[deviceRank].DeviceId,
                    DstMemAddr:  commInitResp.Devices[deviceRank].MinMemAddr,
                },
            },
        })
        if err != nil {
            t.Fatalf("Failed to copy data to GPU %d: %v", deviceRank, err)
        }
    }

    //setup memory addresses
    memAddrs := make(map[uint32]*pb.MemAddr)
    for i := uint32(0); i < uint32(N); i++ {
        memAddrs[i] = commInitResp.Devices[i].MinMemAddr
    }

    //execute NaiveAllReduce
    allReduceResp, err := client.NaiveAllReduce(ctx, &pb.NaiveAllReduceRequest{
        CommId:   commInitResp.CommId,
        Count:    totalBytes,
        Op:       pb.ReduceOp_SUM,
        MemAddrs: memAddrs,
    })
    if err != nil {
        t.Fatalf("NaiveAllReduce failed: %v", err)
    }
    if !allReduceResp.Success {
        t.Fatal("NaiveAllReduce reported failure")
    }

    //verify results from all devices
    expectedSum := 10.0 // 1 + 2 + 3 + 4
    for deviceRank := 0; deviceRank < N; deviceRank++ {
        resp, err := client.Memcpy(ctx, &pb.MemcpyRequest{
            Either: &pb.MemcpyRequest_DeviceToHost{
                DeviceToHost: &pb.MemcpyDeviceToHostRequest{
                    SrcDeviceId: commInitResp.Devices[deviceRank].DeviceId,
                    SrcMemAddr:  commInitResp.Devices[deviceRank].MinMemAddr,
                    NumBytes:    totalBytes,
                },
            },
        })
        if err != nil {
            t.Fatalf("Failed to copy result from GPU %d: %v", deviceRank, err)
        }

        result := make([]float64, vectorSize)
        for i := range result {
            result[i] = math.Float64frombits(binary.LittleEndian.Uint64(
                resp.GetDeviceToHost().DstData[i*8:]))
            if math.Abs(result[i]-expectedSum) > 1e-10 {
                t.Errorf("Device %d, Result[%d] = %f, want %f", 
                    deviceRank, i, result[i], expectedSum)
            }
        }
        
    }
}
func TestAllReducePerformance(t *testing.T) {
    client, _, cleanup := setupTest(t)
    defer cleanup()

    ctx := context.Background()

    //calculate maximum safe data size (leaving some headroom in 4MB device memory)
    maxBytesPerDevice := uint64(3.5 * 1024 * 1024) // 3.5MB to leave some headroom
    bytesPerFloat := 8

    sizes := []struct {
        name string
        vectorSize int
        iterations int
    }{
        {"Tiny", 1000, 5},            // 8KB
        {"Small", 10000, 5},          // 80KB
        {"Medium", 50000, 3},         // 400KB
        {"Large", 100000, 2},         // 800KB
        {"XLarge", 200000, 2},        // 1.6MB
        {"XXLarge", 450000, 1},       
    }

    numDevices := 7

    for _, size := range sizes {
        t.Run(size.name, func(t *testing.T) {
            totalBytes := uint64(size.vectorSize * bytesPerFloat)
            
            // Skip if total data size would exceed device memory
            if totalBytes > maxBytesPerDevice {
                t.Skipf("Skipping %s size - would exceed device memory (size=%d bytes, max=%d bytes)", 
                    size.name, totalBytes, maxBytesPerDevice)
                return
            }

            var naiveTimes []time.Duration
            var ringTimes []time.Duration

            for iter := 0; iter < size.iterations; iter++ {
                // t.Logf("Running %s size iteration %d/%d...", size.name, iter+1, size.iterations)
                
                // Initialize communicator for this iteration
                commInitResp, err := client.CommInit(ctx, &pb.CommInitRequest{NumDevices: uint32(numDevices)})
                if err != nil {
                    t.Fatalf("CommInit failed: %v", err)
                }

                memAddrs := make(map[uint32]*pb.MemAddr)
                for i := uint32(0); i < uint32(numDevices); i++ {
                    memAddrs[i] = commInitResp.Devices[i].MinMemAddr
                }

                //setup test data
                setupData := func() {
                    for deviceRank := 0; deviceRank < numDevices; deviceRank++ {
                        data := make([]byte, totalBytes)
                        value := float64(deviceRank + 1)
                        for i := 0; i < size.vectorSize; i++ {
                            binary.LittleEndian.PutUint64(data[i*8:], math.Float64bits(value))
                        }
                        _, err := client.Memcpy(ctx, &pb.MemcpyRequest{
                            Either: &pb.MemcpyRequest_HostToDevice{
                                HostToDevice: &pb.MemcpyHostToDeviceRequest{
                                    HostSrcData: data,
                                    DstDeviceId: commInitResp.Devices[deviceRank].DeviceId,
                                    DstMemAddr:  commInitResp.Devices[deviceRank].MinMemAddr,
                                },
                            },
                        })
                        if err != nil {
                            t.Fatalf("Failed to copy data to GPU %d: %v", deviceRank, err)
                        }
                    }
                }

                //test naive allreduce
                setupData()
                // t.Logf("Starting Naive AllReduce for size %s...", size.name)
                start := time.Now()
                _, err = client.NaiveAllReduce(ctx, &pb.NaiveAllReduceRequest{
                    CommId:   commInitResp.CommId,
                    Count:    totalBytes,
                    Op:       pb.ReduceOp_SUM,
                    MemAddrs: memAddrs,
                })
                if err != nil {
                    t.Fatalf("NaiveAllReduce failed: %v", err)
                }
                naiveTime := time.Since(start)
                naiveTimes = append(naiveTimes, naiveTime)
                // t.Logf("Naive completed in %v", naiveTime)

                //test ring allreduce
                setupData()
                // t.Logf("Starting Ring AllReduce for size %s...", size.name)
                start = time.Now()
                _, err = client.GroupStart(ctx, &pb.GroupStartRequest{CommId: commInitResp.CommId})
                if err != nil {
                    t.Fatalf("GroupStart failed: %v", err)
                }
                _, err = client.AllReduceRing(ctx, &pb.AllReduceRingRequest{
                    CommId:   commInitResp.CommId,
                    Count:    totalBytes,
                    Op:       pb.ReduceOp_SUM,
                    MemAddrs: memAddrs,
                })
                if err != nil {
                    t.Fatalf("AllReduceRing failed: %v", err)
                }
                _, err = client.GroupEnd(ctx, &pb.GroupEndRequest{CommId: commInitResp.CommId})
                if err != nil {
                    t.Fatalf("GroupEnd failed: %v", err)
                }
                ringTime := time.Since(start)
                ringTimes = append(ringTimes, ringTime)
                // t.Logf("Ring completed in %v", ringTime)

                // Add cooldown between iterations
                time.Sleep(100 * time.Millisecond)
            }

            //calculate average times
            var avgNaive, avgRing time.Duration
            for i := 0; i < size.iterations; i++ {
                avgNaive += naiveTimes[i]
                avgRing += ringTimes[i]
            }
            avgNaive /= time.Duration(size.iterations)
            avgRing /= time.Duration(size.iterations)

            t.Logf("\nPerformance comparison for %s size (%d elements, %.2f MB), averaged over %d iterations:",
                size.name, size.vectorSize, float64(totalBytes)/(1024*1024), size.iterations)
            t.Logf("Naive AllReduce: %v", avgNaive)
            t.Logf("Ring AllReduce:  %v", avgRing)
            t.Logf("Speedup: %.2fx", float64(avgNaive)/float64(avgRing))
        })
    }
}


// Add to pkg/tests/gpu_test.go
// func TestAllReduceWithFailureDetection(t *testing.T) {
//     client, gpuCoordinator, cleanup := setupTest(t)
//     defer cleanup()

//     ctx := context.Background()

//     // Setup parameters
//     N := 4          
//     vectorSize := 8 
//     bytesPerFloat := 8
//     totalBytes := uint64(vectorSize * bytesPerFloat)

//     // Initialize communicator
//     commInitResp, err := client.CommInit(ctx, &pb.CommInitRequest{NumDevices: uint32(N)})
//     if err != nil {
//         t.Fatalf("CommInit failed: %v", err)
//     }

//     // Initialize test data on each GPU
//     for deviceRank := 0; deviceRank < N; deviceRank++ {
//         data := make([]byte, totalBytes)
//         value := float64(deviceRank + 1)
        
//         for i := 0; i < vectorSize; i++ {
//             binary.LittleEndian.PutUint64(data[i*8:], math.Float64bits(value))
//         }

//         _, err := client.Memcpy(ctx, &pb.MemcpyRequest{
//             Either: &pb.MemcpyRequest_HostToDevice{
//                 HostToDevice: &pb.MemcpyHostToDeviceRequest{
//                     HostSrcData: data,
//                     DstDeviceId: commInitResp.Devices[deviceRank].DeviceId,
//                     DstMemAddr:  commInitResp.Devices[deviceRank].MinMemAddr,
//                 },
//             },
//         })
//         if err != nil {
//             t.Fatalf("Failed to copy data to GPU %d: %v", deviceRank, err)
//         }
//     }

//     // Setup memory addresses for AllReduce
//     memAddrs := make(map[uint32]*pb.MemAddr)
//     for i := uint32(0); i < uint32(N); i++ {
//         memAddrs[i] = commInitResp.Devices[i].MinMemAddr
//     }

//     // Test cases
//     testCases := []struct {
//         name string
//         failureScenario func(*testing.T)
//         expectedDevices int
//         shouldSucceed bool
//         getExpectedSum func() float64  // Add this to calculate expected sum based on active devices
//     }{
//         {
//             name: "Single Device Failure During Scatter",
//             failureScenario: func(t *testing.T) {
//                 // Fail device 1 (value 2)
//                 d:= gpuCoordinator.Devices[uint64(1)]
//                 d.SetLastActiveTime(time.Now().Add(-3 * device.HealthCheckTimeout))
//             },
//             expectedDevices: N - 1,
//             shouldSucceed: true,
//             getExpectedSum: func() float64 {
//                 // Sum should be 1 + 3 + 4 = 8 (device 2 failed)
//                 return 8.0
//             },
//         },
//         {
//             name: "Multiple Device Failures",
//             failureScenario: func(t *testing.T) {
//                 // Fail devices 1 and 2 (values 2 and 3)
//                 for id := uint64(1); id <= 2; id++ {
//                     d := gpuCoordinator.Devices[id]
//                     d.SetLastActiveTime(time.Now().Add(-3 * device.HealthCheckTimeout))
//                 }
//             },
//             expectedDevices: N - 2,
//             shouldSucceed: true,
//             getExpectedSum: func() float64 {
//                 // Sum should be 1 + 4 = 5 (devices 2 and 3 failed)
//                 return 5.0
//             },
//         },
//         {
//             name: "Too Many Failures",
//             failureScenario: func(t *testing.T) {
//                 // Fail all but one device
//                 for i := 1; i < N; i++ {
//                     d := gpuCoordinator.Devices[uint64(i)]
//                     d.SetLastActiveTime(time.Now().Add(-3 * device.HealthCheckTimeout))
//                 }
//             },
//             expectedDevices: 1,
//             shouldSucceed: false,
//             getExpectedSum: func() float64 {
//                 return 0 // Operation should fail
//             },
//         },
//     }

//     for _, tc := range testCases {
//         t.Run(tc.name, func(t *testing.T) {
//             // Reset device states
//             for _, device := range gpuCoordinator.Devices {
//                 device.SetLastActiveTime(time.Now())
//             }

//             // Apply failure scenario
//             tc.failureScenario(t)

//             // Start group operation
//             _, err := client.GroupStart(ctx, &pb.GroupStartRequest{CommId: commInitResp.CommId})
//             if err != nil {
//                 t.Fatalf("GroupStart failed: %v", err)
//             }

//             // Execute AllReduce
//             allReduceResp, err := client.AllReduceRing(ctx, &pb.AllReduceRingRequest{
//                 CommId:   commInitResp.CommId,
//                 Count:    totalBytes,
//                 Op:       pb.ReduceOp_SUM,
//                 MemAddrs: memAddrs,
//             })

//             if tc.shouldSucceed {
//                 if err != nil {
//                     t.Errorf("Expected success, got error: %v", err)
//                 }
//                 if !allReduceResp.Success {
//                     t.Error("AllReduce reported failure when success expected")
//                 }

//                 // Verify result from first device (which is always active in our tests)
//                 resp, err := client.Memcpy(ctx, &pb.MemcpyRequest{
//                     Either: &pb.MemcpyRequest_DeviceToHost{
//                         DeviceToHost: &pb.MemcpyDeviceToHostRequest{
//                             SrcDeviceId: commInitResp.Devices[0].DeviceId,
//                             SrcMemAddr:  commInitResp.Devices[0].MinMemAddr,
//                             NumBytes:    totalBytes,
//                         },
//                     },
//                 })
//                 if err != nil {
//                     t.Fatalf("Failed to copy result from GPU: %v", err)
//                 }

//                 // Verify results
//                 expectedSum := tc.getExpectedSum()
//                 result := make([]float64, vectorSize)
//                 for i := range result {
//                     result[i] = math.Float64frombits(binary.LittleEndian.Uint64(
//                         resp.GetDeviceToHost().DstData[i*8:]))
//                     if math.Abs(result[i]-expectedSum) > 1e-10 {
//                         t.Errorf("Result[%d] = %f, want %f", i, result[i], expectedSum)
//                     }
//                 }
//             } else {
//                 if err == nil {
//                     t.Error("Expected failure, got success")
//                 }
//             }

//             // End group operation
//             _, err = client.GroupEnd(ctx, &pb.GroupEndRequest{CommId: commInitResp.CommId})
//             if err != nil {
//                 t.Fatalf("GroupEnd failed: %v", err)
//             }
//         })
//     }
// }

// func TestSimpleAllReduceWithFailure(t *testing.T) {
//     client, gpuCoordinator, cleanup := setupTest(t)
//     defer cleanup()
//     ctx := context.Background()

//     // Initialize with just 3 devices
//     commInitResp, err := client.CommInit(ctx, &pb.CommInitRequest{NumDevices: 3})
//     if err != nil {
//         t.Fatalf("CommInit failed: %v", err)
//     }

//     // Set initial values: [1, 2, 3]
//     for deviceRank := 0; deviceRank < 3; deviceRank++ {
//         value := float64(deviceRank + 1)
//         data := make([]byte, 8) // Just one float64
//         binary.LittleEndian.PutUint64(data, math.Float64bits(value))
        
//         _, err := client.Memcpy(ctx, &pb.MemcpyRequest{
//             Either: &pb.MemcpyRequest_HostToDevice{
//                 HostToDevice: &pb.MemcpyHostToDeviceRequest{
//                     HostSrcData: data,
//                     DstDeviceId: commInitResp.Devices[deviceRank].DeviceId,
//                     DstMemAddr:  commInitResp.Devices[deviceRank].MinMemAddr,
//                 },
//             },
//         })
//         if err != nil {
//             t.Fatalf("Failed to copy data to GPU %d: %v", deviceRank, err)
//         }

//         // Verify initialization
//         resp, err := client.Memcpy(ctx, &pb.MemcpyRequest{
//             Either: &pb.MemcpyRequest_DeviceToHost{
//                 DeviceToHost: &pb.MemcpyDeviceToHostRequest{
//                     SrcDeviceId: commInitResp.Devices[deviceRank].DeviceId,
//                     SrcMemAddr:  commInitResp.Devices[deviceRank].MinMemAddr,
//                     NumBytes:    8,
//                 },
//             },
//         })
//         if err != nil {
//             t.Fatalf("Failed to verify data on GPU %d: %v", deviceRank, err)
//         }
//         initValue := math.Float64frombits(binary.LittleEndian.Uint64(resp.GetDeviceToHost().DstData))
//         log.Printf("Initial value for device %d: %f", deviceRank, initValue)
//     }

//     // Fail device 1
//     gpuCoordinator.Devices[1].SetLastActiveTime(time.Now().Add(-3 * device.HealthCheckTimeout))

//     // Setup memory addresses
//     memAddrs := make(map[uint32]*pb.MemAddr)
//     for i := uint32(0); i < 3; i++ {
//         memAddrs[i] = commInitResp.Devices[i].MinMemAddr
//     }

//     // Execute AllReduce
//     _, err = client.AllReduceRing(ctx, &pb.AllReduceRingRequest{
//         CommId:   commInitResp.CommId,
//         Count:    8, // Just one float64
//         Op:       pb.ReduceOp_SUM,
//         MemAddrs: memAddrs,
//     })
//     if err != nil {
//         t.Fatalf("AllReduce failed: %v", err)
//     }

//     // Verify results from all active devices
//     for deviceRank := 0; deviceRank < 3; deviceRank++ {
//         // Skip failed device
//         if deviceRank == 1 {
//             continue
//         }

//         resp, err := client.Memcpy(ctx, &pb.MemcpyRequest{
//             Either: &pb.MemcpyRequest_DeviceToHost{
//                 DeviceToHost: &pb.MemcpyDeviceToHostRequest{
//                     SrcDeviceId: commInitResp.Devices[deviceRank].DeviceId,
//                     SrcMemAddr:  commInitResp.Devices[deviceRank].MinMemAddr,
//                     NumBytes:    8,
//                 },
//             },
//         })
//         if err != nil {
//             t.Fatalf("Failed to copy result from GPU %d: %v", deviceRank, err)
//         }

//         result := math.Float64frombits(binary.LittleEndian.Uint64(resp.GetDeviceToHost().DstData))
//         expected := 4.0 // 1 + 3
//         log.Printf("Device %d final value: %f (expected %f)", deviceRank, result, expected)
//         if math.Abs(result-expected) > 1e-10 {
//             t.Errorf("Device %d: Got result %f, want %f", deviceRank, result, expected)
//         }
//     }
// }