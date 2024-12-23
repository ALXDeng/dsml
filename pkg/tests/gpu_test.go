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
    // Start the server
    serverAddr, gpuCoordinator, serverCleanup := startServer(t)

    // Setup client options with increased message size
    var opts []grpc.DialOption
    opts = append(opts, 
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithDefaultCallOptions(
            grpc.MaxCallRecvMsgSize(1024*1024*1024), // 1GB
            grpc.MaxCallSendMsgSize(1024*1024*1024), // 1GB
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
    // Find an available port
    listener, err := net.Listen("tcp", ":0")
    if err != nil {
        t.Fatalf("Failed to listen: %v", err)
    }
    
    // Create server with increased message size
    grpcServer := grpc.NewServer(
        grpc.MaxRecvMsgSize(1024*1024*1024), // 1GB
        grpc.MaxSendMsgSize(1024*1024*1024), // 1GB
    )
    
    coord := coordinator.NewGPUCoordinator()
    pb.RegisterGPUCoordinatorServer(grpcServer, coord)

    // Start server in a goroutine
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

    // Initialize communicator
    N := 4
    commInit, err := client.CommInit(ctx, &pb.CommInitRequest{
        NumDevices: uint32(N),
    })
    if err != nil {
        t.Fatalf("Failed to initialize communicator: %v", err)
    }

    // Create test data
    testData := make([]float64, 1024)
    for i := range testData {
        testData[i] = float64(i)
    }

    // Convert to bytes
    dataBytes := make([]byte, len(testData)*8)
    for i, v := range testData {
        binary.LittleEndian.PutUint64(dataBytes[i*8:], math.Float64bits(v))
    }

    // Test copying to each GPU
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

        // Validate data on GPU

        deviceID := commInit.Devices[i].DeviceId.Value
        gpuDevice, exists := gpuCoordinator.Devices[deviceID]
        if !exists {
            t.Errorf("Device with ID %d not found for validation", deviceID)
            continue
        }

        for j := 0; j < len(testData); j++ {
            addr := commInit.Devices[i].MinMemAddr.Value + uint64(j*8)
            actual := math.Float64frombits(binary.LittleEndian.Uint64(gpuDevice.Memory[addr : addr+8]))
            // t.Logf("Data transfer on GPU %d at index %d: got %f, expected %f", i, j, actual, testData[j])
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

    // Initialize communicator
    commInit, err := client.CommInit(ctx, &pb.CommInitRequest{
        NumDevices: 4,
    })
    if err != nil {
        t.Fatalf("Failed to initialize communicator: %v", err)
    }

    // Test GroupStart
    startResp, err := client.GroupStart(ctx, &pb.GroupStartRequest{
        CommId: commInit.CommId,
    })
    if err != nil {
        t.Fatalf("GroupStart failed: %v", err)
    }
    if !startResp.Success {
        t.Error("GroupStart reported failure")
    }

    // Test GroupEnd
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
    
    // Use a longer timeout for this test
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    // Initialize communicator
    N := 4
    commInit, err := client.CommInit(ctx, &pb.CommInitRequest{
        NumDevices: uint32(N),
    })
    if err != nil {
        t.Fatalf("Failed to initialize communicator: %v", err)
    }

    // Give some time for device connections to stabilize
    time.Sleep(100 * time.Millisecond)
    // Prepare memory addresses for AllReduce
    memAddrs := make(map[uint32]*pb.MemAddr)
    for i := uint32(0); i < uint32(N); i++ {
        memAddrs[i] = commInit.Devices[i].MinMemAddr
    }

    // Start group operation
    _, err = client.GroupStart(ctx, &pb.GroupStartRequest{
        CommId: commInit.CommId,
    })
    if err != nil {
        t.Fatalf("GroupStart failed: %v", err)
    }

    // Perform AllReduce
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

    // End group operation
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

    // Initialize communicator
    commInit, err := client.CommInit(ctx, &pb.CommInitRequest{
        NumDevices: 4,
    })
    if err != nil {
        t.Fatalf("Failed to initialize communicator: %v", err)
    }

    // Check initial status
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

    // Initialize communicator
    commInit, err := client.CommInit(ctx, &pb.CommInitRequest{
        NumDevices: 4,
    })
    if err != nil {
        t.Fatalf("Failed to initialize communicator: %v", err)
    }

    // Create test data and copy to GPU
    testData := make([]float64, 1024)
    for i := range testData {
        testData[i] = float64(i)
    }

    // Convert test data to bytes
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

    // Copy data back from GPU
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

func TestAllReduceRing(t *testing.T) {
    client, _, cleanup := setupTest(t)
    defer cleanup()

    ctx := context.Background()

    // Setup parameters
    N := 4          // number of GPUs
    vectorSize := 8 // elements per vector - MUST be divisible by number of GPUs
    bytesPerFloat := 8
    totalBytes := uint64(vectorSize * bytesPerFloat)

    // Initialize communicator
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

        // Copy to GPU
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

    // Initialize memory addresses map for AllReduce
    memAddrs := make(map[uint32]*pb.MemAddr)
    for i := uint32(0); i < uint32(N); i++ {
        memAddrs[i] = commInitResp.Devices[i].MinMemAddr
    }

    // Start group operation
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

    // End group operation
    if _, err := client.GroupEnd(ctx, &pb.GroupEndRequest{CommId: commInitResp.CommId}); err != nil {
        t.Fatalf("GroupEnd failed: %v", err)
    }

    // Wait for operation to complete
    time.Sleep(100 * time.Millisecond)

    // Copy result back from GPU 0 and verify
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

    // Convert result back to float64 slice and verify
    result := make([]float64, vectorSize)
    data := resp.GetDeviceToHost().DstData
    
    // Print the raw bytes for debugging
    // t.Logf("Raw result bytes: %v", data)
    
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

    // Setup parameters
    N := 4          // number of GPUs
    vectorSize := 8 // must be divisible by number of GPUs
    bytesPerFloat := 8
    totalBytes := uint64(vectorSize * bytesPerFloat)

    // Initialize communicator
    commInitResp, err := client.CommInit(ctx, &pb.CommInitRequest{NumDevices: uint32(N)})
    if err != nil {
        t.Fatalf("CommInit failed: %v", err)
    }

    // Initialize each device with its rank+1 as values
    for deviceRank := 0; deviceRank < N; deviceRank++ {
        data := make([]byte, totalBytes)
        value := float64(deviceRank + 1)
        
        for i := 0; i < vectorSize; i++ {
            binary.LittleEndian.PutUint64(data[i*8:], math.Float64bits(value))
        }

        // t.Logf("Initializing device %d with value %f", deviceRank, value)
        
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

    // Setup memory addresses
    memAddrs := make(map[uint32]*pb.MemAddr)
    for i := uint32(0); i < uint32(N); i++ {
        memAddrs[i] = commInitResp.Devices[i].MinMemAddr
    }

    // Execute NaiveAllReduce
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

    // Verify results from all devices
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
        // t.Logf("Device %d results verified", deviceRank)
    }
}
func TestAllReducePerformance(t *testing.T) {
    client, _, cleanup := setupTest(t)
    defer cleanup()

    ctx := context.Background()

    // Calculate maximum safe data size (leaving some headroom in 4MB device memory)
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

    numDevices := 4

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

            // Run multiple iterations for more stable measurements
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

                // Setup test data
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

                // Test Naive AllReduce
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

                // Test Ring AllReduce
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

            // Calculate average times
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