// pkg/coordinator/coordinator.go
package coordinator

import (
	"context"
	"dsml/pkg/device"
	pb "dsml/proto/gpu_sim"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// Communicator represents a group of GPU devices that can communicate with each other
type Communicator struct {
    id      uint64
    devices []pb.GPUDeviceClient
    status  pb.Status
    inGroup bool
}

// GPUCoordinator manages multiple GPU devices and coordinates their operations
type GPUCoordinator struct {
	pb.UnimplementedGPUCoordinatorServer
	Comms     map[uint64]*Communicator  // Map of communicator ID to Communicator
    Devices   map[uint64]*device.GPUDevice
	CommLock  sync.RWMutex
	NextCommID uint64
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
    
    log.Printf("Naive AllReduce: Gathering all data to device 0 (data size: %d bytes)", dataSize)
    
    // Set all devices to non-allgather phase (for reduction behavior)
    for _, device := range c.Devices {
        device.SetAllGatherPhase(false)
    }
    
    // First gather all data to device 0
    for srcRank := 1; srcRank < numDevices; srcRank++ {
        // Begin send from source device
        sendResp, err := comm.devices[srcRank].BeginSend(ctx, &pb.BeginSendRequest{
            SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(srcRank)].Value},
            NumBytes:    dataSize,
            DstRank:    &pb.Rank{Value: 0},
        })
        if err != nil {
            return nil, fmt.Errorf("gather send failed from rank %d: %v", srcRank, err)
        }

        // Device 0 receives and automatically reduces (adds) to existing values
        _, err = comm.devices[0].BeginReceive(ctx, &pb.BeginReceiveRequest{
            StreamId:     sendResp.StreamId,
            RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[0].Value},  // Same destination address for reduction
            NumBytes:    dataSize,
            SrcRank:     &pb.Rank{Value: uint32(srcRank)},
        })
        if err != nil {
            return nil, fmt.Errorf("gather receive failed from rank %d: %v", srcRank, err)
        }

        // Wait for transfer and reduction to complete
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
            time.Sleep(time.Millisecond)
        }
        
        log.Printf("Gathered and reduced data from device %d", srcRank)
    }

    // Switch to broadcast mode (no more reduction needed)
    for _, device := range c.Devices {
        device.SetAllGatherPhase(true)
    }

    log.Printf("Naive AllReduce: Broadcasting reduced result to all devices")
    
    // Broadcast the reduced result from device 0 to all other devices
    for dstRank := 1; dstRank < numDevices; dstRank++ {
        // Begin send from device 0
        sendResp, err := comm.devices[0].BeginSend(ctx, &pb.BeginSendRequest{
            SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[0].Value},
            NumBytes:    dataSize,
            DstRank:    &pb.Rank{Value: uint32(dstRank)},
        })
        if err != nil {
            return nil, fmt.Errorf("broadcast send failed to rank %d: %v", dstRank, err)
        }

        // Destination device receives (no reduction, just copy)
        _, err = comm.devices[dstRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
            StreamId:     sendResp.StreamId,
            RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(dstRank)].Value},
            NumBytes:    dataSize,
            SrcRank:     &pb.Rank{Value: 0},
        })
        if err != nil {
            return nil, fmt.Errorf("broadcast receive failed at rank %d: %v", dstRank, err)
        }

        // Wait for broadcast to complete
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
            time.Sleep(time.Millisecond)
        }
        
        log.Printf("Broadcast complete to device %d", dstRank)
    }

    comm.status = pb.Status_SUCCESS
    return &pb.NaiveAllReduceResponse{Success: true}, nil
}
// CommInit initializes a new communicator with the specified number of devices
func (c *GPUCoordinator) CommInit(ctx context.Context, req *pb.CommInitRequest) (*pb.CommInitResponse, error) {
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



// GetCommStatus returns the status of a communicator
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

// GroupStart begins a group operation
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

// GroupEnd ends a group operation
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

// AllReduceRing implements the ring-allreduce algorithm

func (c *GPUCoordinator) AllReduceRing(ctx context.Context, req *pb.AllReduceRingRequest) (*pb.AllReduceRingResponse, error) {
    c.CommLock.RLock()
    comm, exists := c.Comms[req.CommId]
    c.CommLock.RUnlock()

    if !exists {
        return nil, fmt.Errorf("communicator not found")
    }

    // Reset all devices to scatter-reduce phase
    for _, device := range c.Devices {
        device.SetAllGatherPhase(false)
    }

    // Execute scatter-reduce phase
    if err := c.scatterReducePhase(ctx, comm, req); err != nil {
        comm.status = pb.Status_FAILED
        return nil, fmt.Errorf("scatter-reduce failed: %v", err)
    }

    // Wait for all operations to complete
    time.Sleep(10 * time.Millisecond)

    // Set all devices to all-gather phase
    for _, device := range c.Devices {
        device.SetAllGatherPhase(true)
        // log.Printf("Setting device %d to all-gather phase", device.Rank)
    }

    // Wait for phase change to propagate
    time.Sleep(10 * time.Millisecond)

    // Execute all-gather phase
    if err := c.allGatherPhase(ctx, comm, req); err != nil {
        comm.status = pb.Status_FAILED
        return nil, fmt.Errorf("all-gather failed: %v", err)
    }

    return &pb.AllReduceRingResponse{Success: true}, nil
}



func (c *GPUCoordinator) scatterReducePhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
    numDevices := len(comm.devices)
    chunkSize := req.Count / uint64(numDevices)
    
    // log.Printf("Starting scatter-reduce phase with %d devices, chunk size: %d bytes", numDevices, chunkSize)
    
    // In scatter-reduce phase, each device sends its chunk to next device
    // After N-1 steps, each device has partial sum of one chunk
    for step := 0; step < numDevices-1; step++ {
        // log.Printf("Scatter-reduce step %d/%d", step+1, numDevices-1)
        
        for rank := 0; rank < numDevices; rank++ {
            sendRank := rank
            recvRank := (rank + 1) % numDevices
            
            // Calculate source chunk index
            // In step 0: device i sends chunk i
            // In step 1: device i sends chunk i-1
            // etc.
            chunkIndex := (rank - step + numDevices) % numDevices
            offset := uint64(chunkIndex) * chunkSize
            
            // log.Printf("Step %d: Device %d sending chunk %d to device %d (offset=%d)", 
            //     step, sendRank, chunkIndex, recvRank, offset)
            
            // Begin send
            sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
                SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + offset},
                NumBytes:    chunkSize,
                DstRank:    &pb.Rank{Value: uint32(recvRank)},
            })
            if err != nil {
                return fmt.Errorf("send failed at step %d, rank %d: %v", step, sendRank, err)
            }
            
            // Begin receive - write to same chunk position
            _, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
                StreamId:     sendResp.StreamId,
                RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + offset},
                NumBytes:    chunkSize,
                SrcRank:     &pb.Rank{Value: uint32(sendRank)},
            })
            if err != nil {
                return fmt.Errorf("receive failed at step %d, rank %d: %v", step, recvRank, err)
            }
            
            // Wait for completion
            for {
                statusResp, err := comm.devices[recvRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
                    StreamId: sendResp.StreamId,
                })
                if err != nil {
                    return fmt.Errorf("status check failed: %v", err)
                }
                if statusResp.Status != pb.Status_IN_PROGRESS {
                    break
                }
                time.Sleep(time.Millisecond)
            }
        }
        
        // Small delay between steps to ensure stability
        time.Sleep(5 * time.Millisecond)
    }
    
    return nil
}


func (c *GPUCoordinator) allGatherPhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
    numDevices := len(comm.devices)
    chunkSize := req.Count / uint64(numDevices)
    
    log.Printf("Starting all-gather phase with %d devices", numDevices)
    
    // Track which chunks each device has
    deviceChunks := make(map[int][]int)
    for i := 0; i < numDevices; i++ {
        // Initially device i has its reduced chunk at ((i+1) % N)
        deviceChunks[i] = []int{(i + 1) % numDevices}
        // log.Printf("Device %d starts with reduced chunk %v", i, deviceChunks[i])
    }
    
    // All-gather phase: each device needs to get all chunks
    for step := 0; step < numDevices-1; step++ {
        // log.Printf("All-gather step %d/%d", step+1, numDevices-1)
        
        for rank := 0; rank < numDevices; rank++ {
            sendRank := rank
            recvRank := (rank + 1) % numDevices
            
            // Send each chunk this device has
            for _, chunkId := range deviceChunks[sendRank] {
                offset := uint64(chunkId) * chunkSize
                
                // log.Printf("Step %d: Device %d sending chunk %d (offset %d) to device %d", 
                //     step, sendRank, chunkId, offset, recvRank)
                
                sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
                    SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + offset},
                    NumBytes:    chunkSize,
                    DstRank:    &pb.Rank{Value: uint32(recvRank)},
                })
                if err != nil {
                    return fmt.Errorf("send failed: %v", err)
                }
                
                _, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
                    StreamId:     sendResp.StreamId,
                    RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + offset},
                    NumBytes:    chunkSize,
                    SrcRank:     &pb.Rank{Value: uint32(sendRank)},
                })
                if err != nil {
                    return fmt.Errorf("receive failed: %v", err)
                }
                
                // Wait for completion
                for {
                    statusResp, err := comm.devices[recvRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
                        StreamId: sendResp.StreamId,
                    })
                    if err != nil {
                        return fmt.Errorf("status check failed: %v", err)
                    }
                    if statusResp.Status != pb.Status_IN_PROGRESS {
                        break
                    }
                    time.Sleep(time.Millisecond)
                }
                
                // Update receiver's chunks
                if !contains(deviceChunks[recvRank], chunkId) {
                    deviceChunks[recvRank] = append(deviceChunks[recvRank], chunkId)
                }
            }
        }
        
        // Debug: print chunk distribution
        // for i := 0; i < numDevices; i++ {
        //     // log.Printf("After step %d: Device %d has chunks %v", step, i, deviceChunks[i])
        // }
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
        // Host-to-Device copy
        deviceID := v.HostToDevice.DstDeviceId.Value
        dstAddr := v.HostToDevice.DstMemAddr.Value
        srcData := v.HostToDevice.HostSrcData

        // Locate the target device
        c.CommLock.RLock()
        targetDevice, exists := c.Devices[deviceID]
        c.CommLock.RUnlock()

        if !exists {
            return nil, status.Errorf(codes.NotFound, "Device with ID %d not found", deviceID)
        }

        // Validate memory bounds
        if dstAddr+uint64(len(srcData)) > targetDevice.MaxAddr {
            return nil, status.Errorf(codes.InvalidArgument, "address out of bounds")
        }

        // Copy data
        copy(targetDevice.Memory[dstAddr:], srcData)

        return &pb.MemcpyResponse{
            Either: &pb.MemcpyResponse_HostToDevice{
                HostToDevice: &pb.MemcpyHostToDeviceResponse{
                    Success: true,
                },
            },
        }, nil

    case *pb.MemcpyRequest_DeviceToHost:
        // Device-to-Host copy
        deviceID := v.DeviceToHost.SrcDeviceId.Value
        srcAddr := v.DeviceToHost.SrcMemAddr.Value
        numBytes := v.DeviceToHost.NumBytes

        // Locate the source device
        c.CommLock.RLock()
        sourceDevice, exists := c.Devices[deviceID]
        c.CommLock.RUnlock()

        if !exists {
            return nil, status.Errorf(codes.NotFound, "Device with ID %d not found", deviceID)
        }

        // Validate memory bounds
        if srcAddr+numBytes > sourceDevice.MaxAddr {
            return nil, status.Errorf(codes.InvalidArgument, "address out of bounds")
        }

        // Copy data
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

