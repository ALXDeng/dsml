// pkg/coordinator/coordinator.go
package coordinator

import (
	"context"
	"dsml/pkg/device"
	pb "dsml/proto/gpu_sim"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"net"
	"sort"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

type Communicator1 struct {
    id      uint64
    devices []pb.GPUDeviceClient
    status  pb.Status
    inGroup bool
}
type Communicator struct {
    id       uint64
    devices  []pb.GPUDeviceClient
    status   pb.Status
    inGroup  bool
    failedDevices map[uint32]bool  // Track failed devices by rank
    deviceLock    sync.RWMutex     // Protect device state changes
}



type GPUCoordinator struct {
	pb.UnimplementedGPUCoordinatorServer
	Comms     map[uint64]*Communicator  // Map of communicator ID to Communicator
    Devices   map[uint64]*device.GPUDevice
	CommLock  sync.RWMutex
	NextCommID uint64
}

func (c *GPUCoordinator) monitorDeviceHealth(ctx context.Context, comm *Communicator) {
    ticker := time.NewTicker(device.HealthCheckTimeout / 2)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            for rank, device := range c.Devices {
                if !device.IsHealthy() {
                    comm.deviceLock.Lock()
                    comm.failedDevices[uint32(rank)] = true
                    comm.deviceLock.Unlock()
                }
            }
        }
    }
}

func NewGPUCoordinator() *GPUCoordinator {
	return &GPUCoordinator{
	    Comms:     make(map[uint64]*Communicator),
        Devices:   make(map[uint64]*device.GPUDevice),
        CommLock:  sync.RWMutex{},
		NextCommID: 1,
	}
}

func (c *GPUCoordinator) NaiveAllReduce(ctx context.Context, req *pb.NaiveAllReduceRequest) (*pb.NaiveAllReduceResponse, error) {
    c.CommLock.RLock()
    comm, exists := c.Comms[req.CommId]
    c.CommLock.RUnlock()

    if !exists {
        return nil, fmt.Errorf("communicator not found")
    }

    comm.status = pb.Status_IN_PROGRESS
    numDevices := len(comm.devices)
    dataSize := req.Count
    
    for _, device := range c.Devices {
        device.SetAllGatherPhase(false)
    }
    
    for srcRank := 1; srcRank < numDevices; srcRank++ {

        sendResp, err := comm.devices[srcRank].BeginSend(ctx, &pb.BeginSendRequest{
            SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(srcRank)].Value},
            NumBytes:    dataSize,
            DstRank:    &pb.Rank{Value: 0},
        })
        if err != nil {
            return nil, fmt.Errorf("gather send failed from rank %d: %v", srcRank, err)
        }

        _, err = comm.devices[0].BeginReceive(ctx, &pb.BeginReceiveRequest{
            StreamId:     sendResp.StreamId,
            RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[0].Value},  
            NumBytes:    dataSize,
            SrcRank:     &pb.Rank{Value: uint32(srcRank)},
        })
        if err != nil {
            return nil, fmt.Errorf("gather receive failed from rank %d: %v", srcRank, err)
        }

        for {
            statusResp, err := comm.devices[0].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
                StreamId: sendResp.StreamId,
            })
            if err != nil {
                return nil, fmt.Errorf("status check failed: %v", err)
            }
            if statusResp.Status != pb.Status_IN_PROGRESS {
                break
            }
            time.Sleep(5 * time.Millisecond)
        }
        
    }

    for _, device := range c.Devices {
        device.SetAllGatherPhase(true)
    }

    for dstRank := 1; dstRank < numDevices; dstRank++ {
        
        sendResp, err := comm.devices[0].BeginSend(ctx, &pb.BeginSendRequest{
            SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[0].Value},
            NumBytes:    dataSize,
            DstRank:    &pb.Rank{Value: uint32(dstRank)},
        })
        if err != nil {
            return nil, fmt.Errorf("broadcast send failed to rank %d: %v", dstRank, err)
        }

        _, err = comm.devices[dstRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
            StreamId:     sendResp.StreamId,
            RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(dstRank)].Value},
            NumBytes:    dataSize,
            SrcRank:     &pb.Rank{Value: 0},
        })
        if err != nil {
            return nil, fmt.Errorf("broadcast receive failed at rank %d: %v", dstRank, err)
        }

        for {
            statusResp, err := comm.devices[dstRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
                StreamId: sendResp.StreamId,
            })
            if err != nil {
                return nil, fmt.Errorf("status check failed: %v", err)
            }
            if statusResp.Status != pb.Status_IN_PROGRESS {
                break
            }
            time.Sleep(5 *time.Millisecond)
        }
        
    }

    comm.status = pb.Status_SUCCESS
    return &pb.NaiveAllReduceResponse{Success: true}, nil
}

func (c *GPUCoordinator) CommInit1(ctx context.Context, req *pb.CommInitRequest) (*pb.CommInitResponse, error) {
    if req.NumDevices < 1 {
        return nil, status.Error(codes.InvalidArgument, "number of devices must be positive")
    }

    devices := make([]pb.GPUDeviceClient, req.NumDevices)
    deviceMetas := make([]*pb.DeviceMetadata, req.NumDevices)

    for i := uint32(0); i < req.NumDevices; i++ {
        deviceServer := grpc.NewServer()
        deviceImpl := device.NewGPUDevice(1024*1024, i) // Simulated device
        pb.RegisterGPUDeviceServer(deviceServer, deviceImpl)

        lis, err := net.Listen("tcp", ":0")
        if err != nil {
            return nil, fmt.Errorf("failed to start device %d: %v", i, err)
        }

        go func(index uint32) {
            if err := deviceServer.Serve(lis); err != nil {
                // log.Printf("Device %d server failed: %v", index, err)
            }
        }(i)

        conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
        if err != nil {
            return nil, fmt.Errorf("failed to connect to device %d: %v", i, err)
        }

        devices[i] = pb.NewGPUDeviceClient(conn)
        deviceMetas[i] = &pb.DeviceMetadata{
            DeviceId:    &pb.DeviceId{Value: uint64(i)},
            MinMemAddr:  &pb.MemAddr{Value: 0},
            MaxMemAddr:  &pb.MemAddr{Value: 1024 * 1024},
        }

        // Add the device to the Devices map
        c.Devices[uint64(i)] = deviceImpl
    }

    c.CommLock.Lock()
    commId := c.NextCommID
    c.NextCommID++
    comm := &Communicator{
        id:      commId,
        devices: devices,
        status:  pb.Status_IN_PROGRESS,
        inGroup: false,
    }
    c.Comms[commId] = comm
    c.CommLock.Unlock()

    return &pb.CommInitResponse{
        Success: true,
        CommId:  commId,
        Devices: deviceMetas,
    }, nil
}

func (c *GPUCoordinator) CommInit(ctx context.Context, req *pb.CommInitRequest) (*pb.CommInitResponse, error) {
    if req.NumDevices < 1 {
        return nil, status.Error(codes.InvalidArgument, "number of devices must be positive")
    }

    devices := make([]pb.GPUDeviceClient, req.NumDevices)
    deviceMetas := make([]*pb.DeviceMetadata, req.NumDevices)

    deviceMemory := uint64(4 * 1024 * 1024) // 4MB per device

    for i := uint32(0); i < req.NumDevices; i++ {
        deviceServer := grpc.NewServer()
        deviceImpl := device.NewGPUDevice(deviceMemory, i)
        pb.RegisterGPUDeviceServer(deviceServer, deviceImpl)

        lis, err := net.Listen("tcp", ":0")
        if err != nil {
            return nil, fmt.Errorf("failed to start device %d: %v", i, err)
        }

        go func(index uint32) {
            if err := deviceServer.Serve(lis); err != nil {
                log.Printf("Device %d server failed: %v", index, err)
            }
        }(i)

        conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
        if err != nil {
            return nil, fmt.Errorf("failed to connect to device %d: %v", i, err)
        }

        devices[i] = pb.NewGPUDeviceClient(conn)
        deviceMetas[i] = &pb.DeviceMetadata{
            DeviceId:    &pb.DeviceId{Value: uint64(i)},
            MinMemAddr:  &pb.MemAddr{Value: 0},
            MaxMemAddr:  &pb.MemAddr{Value: deviceMemory},
        }

        c.Devices[uint64(i)] = deviceImpl
    }

    c.CommLock.Lock()
    commId := c.NextCommID
    c.NextCommID++
    comm := &Communicator{
        id:            commId,
        devices:       devices,
        status:        pb.Status_IN_PROGRESS,
        inGroup:       false,
        failedDevices: make(map[uint32]bool), // Initialize the map
        deviceLock:    sync.RWMutex{},
    }
    c.Comms[commId] = comm
    c.CommLock.Unlock()

    return &pb.CommInitResponse{
        Success: true,
        CommId:  commId,
        Devices: deviceMetas,
    }, nil
}


func (c *GPUCoordinator) GetCommStatus(ctx context.Context, req *pb.GetCommStatusRequest) (*pb.GetCommStatusResponse, error) {
    c.CommLock.RLock()
    comm, exists := c.Comms[req.CommId]
    c.CommLock.RUnlock()

    if !exists {
        return nil, status.Errorf(codes.NotFound, "communicator not found")
    }

    return &pb.GetCommStatusResponse{
        Status: comm.status,
    }, nil
}

func (c *GPUCoordinator) GroupStart(ctx context.Context, req *pb.GroupStartRequest) (*pb.GroupStartResponse, error) {
	c.CommLock.Lock()
	defer c.CommLock.Unlock()

	comm, exists := c.Comms[req.CommId]
	if !exists {
		return nil, status.Error(codes.NotFound, "communicator not found")
	}

	if comm.inGroup {
		return nil, status.Error(codes.FailedPrecondition, "group operation already in progress")
	}

	comm.inGroup = true
	return &pb.GroupStartResponse{Success: true}, nil
}

func (c *GPUCoordinator) GroupEnd(ctx context.Context, req *pb.GroupEndRequest) (*pb.GroupEndResponse, error) {
	c.CommLock.Lock()
	defer c.CommLock.Unlock()

	comm, exists := c.Comms[req.CommId]
	if !exists {
		return nil, status.Error(codes.NotFound, "communicator not found")
	}

	if !comm.inGroup {
		return nil, status.Error(codes.FailedPrecondition, "no group operation in progress")
	}

	comm.inGroup = false
	return &pb.GroupEndResponse{Success: true}, nil
}




func (c *GPUCoordinator) AllReduceRing(ctx context.Context, req *pb.AllReduceRingRequest) (*pb.AllReduceRingResponse, error) {
    c.CommLock.RLock()
    comm, exists := c.Comms[req.CommId]
    c.CommLock.RUnlock()

    if !exists {
        return nil, fmt.Errorf("communicator not found")
    }

    // Get active devices and verify we have enough
    activeDevices := make(map[uint32]bool)
    activeRanks := make([]uint32, 0)
    for rank := uint32(0); rank < uint32(len(comm.devices)); rank++ {
        if c.Devices[uint64(rank)].IsDeviceHealthy() {
            activeDevices[rank] = true
            activeRanks = append(activeRanks, rank)
        }
    }
    
    numActive := len(activeRanks)
    if numActive < 2 {
        comm.status = pb.Status_FAILED
        return nil, fmt.Errorf("insufficient active devices (%d) to continue", numActive)
    }

    // Sort ranks for consistent ordering
    sort.Slice(activeRanks, func(i, j int) bool { return activeRanks[i] < activeRanks[j] })

    // Initialize data on active devices
    chunkSize := req.Count / uint64(numActive)
    for _, rank := range activeRanks {
        device := c.Devices[uint64(rank)]
        device.SetAllGatherPhase(false)
        
        // Copy data to proper positions based on new device count
        data := make([]byte, req.Count)
        copy(data, device.Memory[req.MemAddrs[rank].Value:req.MemAddrs[rank].Value+req.Count])
        for i := uint64(0); i < uint64(numActive); i++ {
            destOffset := i * chunkSize
            srcOffset := destOffset
            copy(device.Memory[req.MemAddrs[rank].Value+destOffset:], data[srcOffset:srcOffset+chunkSize])
        }
    }

    // Execute scatter-reduce phase
    if err := c.scatterReducePhase(ctx, comm, req, activeRanks); err != nil {
        comm.status = pb.Status_FAILED
        return nil, fmt.Errorf("scatter-reduce failed: %v", err)
    }

    // Switch to all-gather phase
    for _, rank := range activeRanks {
        c.Devices[uint64(rank)].SetAllGatherPhase(true)
    }

    // Execute all-gather phase
    if err := c.allGatherPhase(ctx, comm, req, activeRanks); err != nil {
        comm.status = pb.Status_FAILED
        return nil, fmt.Errorf("all-gather failed: %v", err)
    }

    return &pb.AllReduceRingResponse{Success: true}, nil
}
func (c *GPUCoordinator) AllReduceRing_logging(ctx context.Context, req *pb.AllReduceRingRequest) (*pb.AllReduceRingResponse, error) {
    c.CommLock.RLock()
    comm, exists := c.Comms[req.CommId]
    c.CommLock.RUnlock()

    if !exists {
        return nil, fmt.Errorf("communicator not found")
    }

    // Get active devices and verify we have enough
    activeRanks := make([]uint32, 0)
    log.Printf("Initial values:")
    for rank := uint32(0); rank < uint32(len(comm.devices)); rank++ {
        device := c.Devices[uint64(rank)]
        val := math.Float64frombits(binary.LittleEndian.Uint64(device.Memory[req.MemAddrs[rank].Value:]))
        log.Printf("Device %d value: %f", rank, val)
        
        if device.IsDeviceHealthy() {
            activeRanks = append(activeRanks, rank)
            log.Printf("Device %d is active", rank)
        } else {
            log.Printf("Device %d is failed", rank)
        }
    }
    
    log.Printf("Active ranks: %v", activeRanks)

    if len(activeRanks) < 2 {
        comm.status = pb.Status_FAILED
        return nil, fmt.Errorf("insufficient active devices (%d) to continue", len(activeRanks))
    }

    // Initialize active devices
    for _, rank := range activeRanks {
        device := c.Devices[uint64(rank)]
        device.SetAllGatherPhase(false)
    }

    // Execute scatter-reduce phase
    if err := c.scatterReducePhase(ctx, comm, req, activeRanks); err != nil {
        comm.status = pb.Status_FAILED
        return nil, fmt.Errorf("scatter-reduce failed: %v", err)
    }

    log.Printf("After scatter-reduce:")
    for _, rank := range activeRanks {
        device := c.Devices[uint64(rank)]
        val := math.Float64frombits(binary.LittleEndian.Uint64(device.Memory[req.MemAddrs[rank].Value:]))
        log.Printf("Device %d value: %f", rank, val)
    }

    // Switch to all-gather phase
    for _, rank := range activeRanks {
        device := c.Devices[uint64(rank)]
        device.SetAllGatherPhase(true)
    }

    // Execute all-gather phase
    if err := c.allGatherPhase(ctx, comm, req, activeRanks); err != nil {
        comm.status = pb.Status_FAILED
        return nil, fmt.Errorf("all-gather failed: %v", err)
    }

    log.Printf("Final values:")
    for _, rank := range activeRanks {
        device := c.Devices[uint64(rank)]
        val := math.Float64frombits(binary.LittleEndian.Uint64(device.Memory[req.MemAddrs[rank].Value:]))
        log.Printf("Device %d value: %f", rank, val)
    }

    return &pb.AllReduceRingResponse{Success: true}, nil
}


func (c *GPUCoordinator) handleDeviceFailure(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) (*pb.AllReduceRingResponse, error) {
    comm.deviceLock.RLock()
    failedDevices := make(map[uint32]bool)
    for rank, failed := range comm.failedDevices {
        failedDevices[rank] = failed
    }
    comm.deviceLock.RUnlock()

    // Create new communicator without failed devices
    activeDevices := make([]pb.GPUDeviceClient, 0)
    activeMemAddrs := make(map[uint32]*pb.MemAddr)
    
    for rank, device := range comm.devices {
        if !failedDevices[uint32(rank)] {
            activeDevices = append(activeDevices, device)
            activeMemAddrs[uint32(rank)] = req.MemAddrs[uint32(rank)]
        }
    }

    if len(activeDevices) < 2 {
        return nil, fmt.Errorf("insufficient active devices to continue")
    }

    // Create new request with only active devices
    newReq := &pb.AllReduceRingRequest{
        CommId:   req.CommId,
        Count:    req.Count,
        Op:       req.Op,
        MemAddrs: activeMemAddrs,
    }

    // Restart operation with remaining devices
    return c.AllReduceRing(ctx, newReq)
}
func (c *GPUCoordinator) scatterReducePhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest, activeRanks []uint32) error {
    numActive := len(activeRanks)
    chunkSize := req.Count / uint64(numActive)
    
    for step := 0; step < numActive-1; step++ {
        var wg sync.WaitGroup
        errChan := make(chan error, 1)

        for i := range activeRanks {
            wg.Add(1)
            go func(idx int) {
                defer wg.Done()

                srcRank := activeRanks[idx]
                dstRank := activeRanks[(idx+1)%numActive]
                chunkIndex := (idx - step + numActive) % numActive
                offset := uint64(chunkIndex) * chunkSize

                sendResp, err := comm.devices[srcRank].BeginSend(ctx, &pb.BeginSendRequest{
                    SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[srcRank].Value + offset},
                    NumBytes:    chunkSize,
                    DstRank:    &pb.Rank{Value: dstRank},
                })
                if err != nil {
                    select {
                    case errChan <- err:
                    default:
                    }
                    return
                }

                _, err = comm.devices[dstRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
                    StreamId:     sendResp.StreamId,
                    RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[dstRank].Value + offset},
                    NumBytes:    chunkSize,
                    SrcRank:     &pb.Rank{Value: srcRank},
                })
                if err != nil {
                    select {
                    case errChan <- err:
                    default:
                    }
                    return
                }

                for {
                    status, err := comm.devices[dstRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
                        StreamId: sendResp.StreamId,
                    })
                    if err != nil {
                        select {
                        case errChan <- err:
                        default:
                        }
                        return
                    }
                    if status.Status != pb.Status_IN_PROGRESS {
                        break
                    }
                    // time.Sleep(100 * time.Microsecond)
                }
            }(i)
        }

        wg.Wait()
        
        select {
        case err := <-errChan:
            return err
        default:
        }
    }
    return nil
}

func (c *GPUCoordinator) allGatherPhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest, activeRanks []uint32) error {
    numActive := len(activeRanks)
    chunkSize := req.Count / uint64(numActive)
    
    for step := 0; step < numActive-1; step++ {
        var wg sync.WaitGroup
        errChan := make(chan error, 1)

        for i := range activeRanks {
            wg.Add(1)
            go func(idx int) {
                defer wg.Done()

                srcRank := activeRanks[idx]
                dstRank := activeRanks[(idx+1)%numActive]
                chunkIndex := (idx - step + 1 + numActive) % numActive
                offset := uint64(chunkIndex) * chunkSize

                sendResp, err := comm.devices[srcRank].BeginSend(ctx, &pb.BeginSendRequest{
                    SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[srcRank].Value + offset},
                    NumBytes:    chunkSize,
                    DstRank:    &pb.Rank{Value: dstRank},
                })
                if err != nil {
                    select {
                    case errChan <- err:
                    default:
                    }
                    return
                }

                _, err = comm.devices[dstRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
                    StreamId:     sendResp.StreamId,
                    RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[dstRank].Value + offset},
                    NumBytes:    chunkSize,
                    SrcRank:     &pb.Rank{Value: srcRank},
                })
                if err != nil {
                    select {
                    case errChan <- err:
                    default:
                    }
                    return
                }

                for {
                    status, err := comm.devices[dstRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
                        StreamId: sendResp.StreamId,
                    })
                    if err != nil {
                        select {
                        case errChan <- err:
                        default:
                        }
                        return
                    }
                    if status.Status != pb.Status_IN_PROGRESS {
                        break
                    }
                    // time.Sleep(100 * time.Microsecond)
                }
            }(i)
        }

        wg.Wait()
        
        select {
        case err := <-errChan:
            return err
        default:
        }
    }
    return nil
}

func contains(slice []int, val int) bool {
    for _, item := range slice {
        if item == val {
            return true
        }
    }
    return false
}

func (c *GPUCoordinator) Memcpy(ctx context.Context, req *pb.MemcpyRequest) (*pb.MemcpyResponse, error) {
    switch v := req.Either.(type) {
    case *pb.MemcpyRequest_HostToDevice:
        deviceID := v.HostToDevice.DstDeviceId.Value
        dstAddr := v.HostToDevice.DstMemAddr.Value
        srcData := v.HostToDevice.HostSrcData

        c.CommLock.RLock()
        targetDevice, exists := c.Devices[deviceID]
        c.CommLock.RUnlock()

        if !exists {
            return nil, status.Errorf(codes.NotFound, "Device with ID %d not found", deviceID)
        }

        if dstAddr+uint64(len(srcData)) > targetDevice.MaxAddr {
            return nil, status.Errorf(codes.InvalidArgument, "address out of bounds")
        }

        copy(targetDevice.Memory[dstAddr:], srcData)

        return &pb.MemcpyResponse{
            Either: &pb.MemcpyResponse_HostToDevice{
                HostToDevice: &pb.MemcpyHostToDeviceResponse{
                    Success: true,
                },
            },
        }, nil

    case *pb.MemcpyRequest_DeviceToHost:
        deviceID := v.DeviceToHost.SrcDeviceId.Value
        srcAddr := v.DeviceToHost.SrcMemAddr.Value
        numBytes := v.DeviceToHost.NumBytes

        c.CommLock.RLock()
        sourceDevice, exists := c.Devices[deviceID]
        c.CommLock.RUnlock()

        if !exists {
            return nil, status.Errorf(codes.NotFound, "Device with ID %d not found", deviceID)
        }

        if srcAddr+numBytes > sourceDevice.MaxAddr {
            return nil, status.Errorf(codes.InvalidArgument, "address out of bounds")
        }

        dstData := make([]byte, numBytes)
        copy(dstData, sourceDevice.Memory[srcAddr:srcAddr+numBytes])

        return &pb.MemcpyResponse{
            Either: &pb.MemcpyResponse_DeviceToHost{
                DeviceToHost: &pb.MemcpyDeviceToHostResponse{
                    DstData: dstData,
                },
            },
        }, nil

    default:
        return nil, status.Errorf(codes.InvalidArgument, "Invalid memcpy request type")
    }
}

