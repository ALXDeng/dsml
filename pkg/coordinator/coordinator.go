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

func (c *GPUCoordinator) AllReduceRing1(ctx context.Context, req *pb.AllReduceRingRequest) (*pb.AllReduceRingResponse, error) {
    c.CommLock.RLock()
    comm, exists := c.Comms[req.CommId]
    c.CommLock.RUnlock()

    if !exists {
        return nil, fmt.Errorf("communicator not found")
    }

    // Set status to IN_PROGRESS at start
    comm.status = pb.Status_IN_PROGRESS

    // Execute scatter-reduce phase
    if err := c.scatterReducePhase(ctx, comm, req); err != nil {
        comm.status = pb.Status_FAILED
        return nil, err
    }

    // Execute all-gather phase
    if err := c.allGatherPhase(ctx, comm, req); err != nil {
        comm.status = pb.Status_FAILED
        return nil, err
    }

    // Set status to SUCCESS when done
    comm.status = pb.Status_SUCCESS

    return &pb.AllReduceRingResponse{
        Success: true,
    }, nil
}

func (c *GPUCoordinator) AllReduceRing(ctx context.Context, req *pb.AllReduceRingRequest) (*pb.AllReduceRingResponse, error) {
    c.CommLock.RLock()
    comm, exists := c.Comms[req.CommId]
    c.CommLock.RUnlock()

    if !exists {
        return nil, fmt.Errorf("communicator not found")
    }

    // Set initial status
    comm.status = pb.Status_IN_PROGRESS

    // Set all devices to scatter-reduce phase
    for _, device := range c.Devices {
        device.SetAllGatherPhase(false)
    }

    // Execute scatter-reduce phase
    if err := c.scatterReducePhase(ctx, comm, req); err != nil {
        comm.status = pb.Status_FAILED
        return nil, fmt.Errorf("scatter-reduce failed: %v", err)
    }

    // Set all devices to all-gather phase
    for _, device := range c.Devices {
        device.SetAllGatherPhase(true)
    }

    // Execute all-gather phase
    if err := c.allGatherPhase(ctx, comm, req); err != nil {
        comm.status = pb.Status_FAILED
        return nil, fmt.Errorf("all-gather failed: %v", err)
    }

    // Set final status
    comm.status = pb.Status_SUCCESS

    return &pb.AllReduceRingResponse{
        Success: true,
    }, nil
}

// scatterReducePhase implements the scatter-reduce phase of ring-allreduce
// func (c *GPUCoordinator) scatterReducePhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
//     if comm == nil || comm.devices == nil {
//         return fmt.Errorf("invalid communicator")
//     }

//     numDevices := len(comm.devices)
//     if numDevices == 0 {
//         return fmt.Errorf("no devices in communicator")
//     }

//     chunkSize := req.Count / uint64(numDevices)

//     for step := 0; step < numDevices-1; step++ {
//         for rank := 0; rank < numDevices; rank++ {
//             sendRank := rank
//             recvRank := (rank + 1) % numDevices

//             if comm.devices[sendRank] == nil || comm.devices[recvRank] == nil {
//                 return fmt.Errorf("device connection not initialized")
//             }

// 			// Calculate memory offsets for this step
// 			sendOffset := uint64(((rank - step + numDevices) % numDevices)) * chunkSize
// 			recvOffset := uint64(((rank - step - 1 + numDevices) % numDevices)) * chunkSize

// 			// Begin send operation
// 			sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
// 				SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + sendOffset},
// 				NumBytes:    chunkSize,
// 				DstRank:    &pb.Rank{Value: uint32(recvRank)},
// 			})
// 			if err != nil {
// 				return fmt.Errorf("failed to begin send: %v", err)
// 			}

// 			// Begin receive operation
// 			_, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
// 				StreamId:     sendResp.StreamId,
// 				RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + recvOffset},
// 				NumBytes:    chunkSize,
// 				SrcRank:     &pb.Rank{Value: uint32(sendRank)},
// 			})
// 			if err != nil {
// 				return fmt.Errorf("failed to begin receive: %v", err)
// 			}

// 			// Wait for completion
// 			for {
// 				statusResp, err := comm.devices[recvRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
// 					StreamId: sendResp.StreamId,
// 				})
// 				if err != nil {
// 					return fmt.Errorf("failed to get stream status: %v", err)
// 				}
// 				if statusResp.Status != pb.Status_IN_PROGRESS {
// 					break
// 				}
// 			}
// 		}
// 	}
// 	return nil
// }

// func (c *GPUCoordinator) scatterReducePhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
//     numDevices := len(comm.devices)
//     chunkSize := req.Count / uint64(numDevices)

//     for step := 0; step < numDevices-1; step++ {
//         for rank := 0; rank < numDevices; rank++ {
//             sendRank := rank
//             recvRank := (rank + 1) % numDevices

//             // Calculate memory offsets
//             sendOffset := uint64(((rank - step + numDevices) % numDevices)) * chunkSize
//             recvOffset := uint64(((rank - step - 1 + numDevices) % numDevices)) * chunkSize

//             // Begin send operation
//             sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
//                 SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + sendOffset},
//                 NumBytes:    chunkSize,
//                 DstRank:    &pb.Rank{Value: uint32(recvRank)},
//             })
//             if err != nil {
//                 return fmt.Errorf("failed to begin send: %v", err)
//             }

//             // Add a small delay to ensure the stream is registered
//             time.Sleep(10 * time.Millisecond)

//             // Begin receive operation
//             _, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
//                 StreamId:     sendResp.StreamId,
//                 RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + recvOffset},
//                 NumBytes:    chunkSize,
//                 SrcRank:     &pb.Rank{Value: uint32(sendRank)},
//             })
//             if err != nil {
//                 return fmt.Errorf("failed to begin receive: %v", err)
//             }

//             // Wait for completion
//             for {
//                 statusResp, err := comm.devices[recvRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
//                     StreamId: sendResp.StreamId,
//                 })
//                 if err != nil {
//                     return fmt.Errorf("failed to get stream status: %v", err)
//                 }
//                 if statusResp.Status != pb.Status_IN_PROGRESS {
//                     break
//                 }
//                 time.Sleep(10 * time.Millisecond)
//             }
//         }
//     }
//     return nil
// }
// func (c *GPUCoordinator) scatterReducePhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
//     numDevices := len(comm.devices)
//     chunkSize := req.Count / uint64(numDevices)

//     for step := 0; step < numDevices-1; step++ {
//         for rank := 0; rank < numDevices; rank++ {
//             sendRank := rank
//             recvRank := (rank + 1) % numDevices

//             // Calculate memory offsets
//             sendOffset := uint64(((rank - step + numDevices) % numDevices)) * chunkSize
//             recvOffset := uint64(((rank - step - 1 + numDevices) % numDevices)) * chunkSize

//             // Begin send operation
//             sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
//                 SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + sendOffset},
//                 NumBytes:    chunkSize,
//                 DstRank:    &pb.Rank{Value: uint32(recvRank)},
//             })
//             if err != nil {
//                 return fmt.Errorf("failed to begin send: %v", err)
//             }

//             // Add a small delay to ensure the stream is registered
//             time.Sleep(10 * time.Millisecond)

//             // Begin receive operation
//             _, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
//                 StreamId:     sendResp.StreamId,
//                 RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + recvOffset},
//                 NumBytes:    chunkSize,
//                 SrcRank:     &pb.Rank{Value: uint32(sendRank)},
//             })
//             if err != nil {
//                 return fmt.Errorf("failed to begin receive: %v", err)
//             }

//             // Wait for completion
//             for {
//                 statusResp, err := comm.devices[recvRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
//                     StreamId: sendResp.StreamId,
//                 })
//                 if err != nil {
//                     return fmt.Errorf("failed to get stream status: %v", err)
//                 }
//                 if statusResp.Status != pb.Status_IN_PROGRESS {
//                     break
//                 }
//                 time.Sleep(10 * time.Millisecond)
//             }
//         }
//     }
//     return nil
// }

func (c *GPUCoordinator) scatterReducePhase2(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
    numDevices := len(comm.devices)
    chunkSize := req.Count / uint64(numDevices)
    
    log.Printf("Starting scatter-reduce phase with %d devices, chunk size: %d bytes", numDevices, chunkSize)
    
    // Scatter-reduce phase
    for step := 0; step < numDevices-1; step++ {
        log.Printf("Scatter-reduce step %d/%d", step+1, numDevices-1)
        
        var wg sync.WaitGroup
        errChan := make(chan error, numDevices)
        
        for rank := 0; rank < numDevices; rank++ {
            wg.Add(1)
            go func(rank int) {
                defer wg.Done()
                
                sendRank := rank
                recvRank := (rank + 1) % numDevices
                
                // Calculate offsets based on ring algorithm
                sendChunk := (rank - step + numDevices) % numDevices
                recvChunk := (rank - step - 1 + numDevices) % numDevices
                
                sendOffset := uint64(sendChunk) * chunkSize
                recvOffset := uint64(recvChunk) * chunkSize
                
                log.Printf("Rank %d sending chunk %d (offset %d) to rank %d, receiving at chunk %d (offset %d)",
                    sendRank, sendChunk, sendOffset, recvRank, recvChunk, recvOffset)
                
                // Begin send
                sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
                    SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + sendOffset},
                    NumBytes:    chunkSize,
                    DstRank:    &pb.Rank{Value: uint32(recvRank)},
                })
                if err != nil {
                    errChan <- fmt.Errorf("send failed at step %d, rank %d: %v", step, sendRank, err)
                    return
                }
                
                // Begin receive
                _, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
                    StreamId:     sendResp.StreamId,
                    RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + recvOffset},
                    NumBytes:    chunkSize,
                    SrcRank:     &pb.Rank{Value: uint32(sendRank)},
                })
                if err != nil {
                    errChan <- fmt.Errorf("receive failed at step %d, rank %d: %v", step, recvRank, err)
                    return
                }
                
                // Wait for completion
                for {
                    statusResp, err := comm.devices[recvRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
                        StreamId: sendResp.StreamId,
                    })
                    if err != nil {
                        errChan <- fmt.Errorf("status check failed: %v", err)
                        return
                    }
                    if statusResp.Status != pb.Status_IN_PROGRESS {
                        if statusResp.Status == pb.Status_FAILED {
                            errChan <- fmt.Errorf("operation failed at step %d between ranks %d->%d", 
                                step, sendRank, recvRank)
                            return
                        }
                        break
                    }
                    time.Sleep(1 * time.Millisecond)
                }
                
                log.Printf("Completed transfer step %d: rank %d -> rank %d", step, sendRank, recvRank)
            }(rank)
        }
        
        // Wait for all transfers in this step
        go func() {
            wg.Wait()
            close(errChan)
        }()
        
        // Check for errors
        for err := range errChan {
            if err != nil {
                return fmt.Errorf("scatter-reduce failed: %v", err)
            }
        }
        
        // Small delay between steps to ensure stability
        time.Sleep(10 * time.Millisecond)
    }
    
    return nil
}
func (c *GPUCoordinator) scatterReducePhase3(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
    numDevices := len(comm.devices)
    chunkSize := req.Count / uint64(numDevices)
    
    log.Printf("Starting scatter-reduce phase with %d devices, chunk size: %d bytes", numDevices, chunkSize)
    
    // In the scatter-reduce phase:
    // Step 0: rank i sends chunk i to rank i+1
    // Step 1: rank i sends chunk (i-1) to rank i+1
    // Step 2: rank i sends chunk (i-2) to rank i+1
    // etc.
    
    for step := 0; step < numDevices-1; step++ {
        log.Printf("Scatter-reduce step %d/%d", step+1, numDevices-1)
        
        for rank := 0; rank < numDevices; rank++ {
            srcRank := rank
            dstRank := (rank + 1) % numDevices
            
            // Calculate which chunk to send
            chunkIdx := (rank - step + numDevices) % numDevices
            
            // Calculate memory offsets
            srcOffset := uint64(chunkIdx) * chunkSize
            dstOffset := srcOffset  // We write to the same offset in the destination
            
            log.Printf("Step %d: Rank %d sending chunk %d (offset %d) to rank %d", 
                step, srcRank, chunkIdx, srcOffset, dstRank)
            
            // Begin send
            sendResp, err := comm.devices[srcRank].BeginSend(ctx, &pb.BeginSendRequest{
                SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(srcRank)].Value + srcOffset},
                NumBytes:    chunkSize,
                DstRank:    &pb.Rank{Value: uint32(dstRank)},
            })
            if err != nil {
                return fmt.Errorf("send failed at step %d, rank %d: %v", step, srcRank, err)
            }
            
            // Begin receive
            _, err = comm.devices[dstRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
                StreamId:     sendResp.StreamId,
                RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(dstRank)].Value + dstOffset},
                NumBytes:    chunkSize,
                SrcRank:     &pb.Rank{Value: uint32(srcRank)},
            })
            if err != nil {
                return fmt.Errorf("receive failed at step %d, rank %d: %v", step, dstRank, err)
            }
            
            // Wait for completion
            for {
                statusResp, err := comm.devices[dstRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
                    StreamId: sendResp.StreamId,
                })
                if err != nil {
                    return fmt.Errorf("status check failed: %v", err)
                }
                if statusResp.Status != pb.Status_IN_PROGRESS {
                    break
                }
                time.Sleep(1 * time.Millisecond)
            }
        }
        time.Sleep(10 * time.Millisecond)  // Small delay between steps
    }
    
    return nil
}

func (c *GPUCoordinator) scatterReducePhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
    numDevices := len(comm.devices)
    chunkSize := req.Count / uint64(numDevices)
    
    log.Printf("Starting scatter-reduce phase with %d devices, chunk size: %d bytes", numDevices, chunkSize)
    
    // In scatter-reduce phase, each device sends its chunk to next device
    // After N-1 steps, each device has partial sum of one chunk
    for step := 0; step < numDevices-1; step++ {
        log.Printf("Scatter-reduce step %d/%d", step+1, numDevices-1)
        
        for rank := 0; rank < numDevices; rank++ {
            sendRank := rank
            recvRank := (rank + 1) % numDevices
            
            // Calculate source chunk index
            // In step 0: device i sends chunk i
            // In step 1: device i sends chunk i-1
            // etc.
            chunkIndex := (rank - step + numDevices) % numDevices
            offset := uint64(chunkIndex) * chunkSize
            
            log.Printf("Step %d: Device %d sending chunk %d to device %d (offset=%d)", 
                step, sendRank, chunkIndex, recvRank, offset)
            
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

// allGatherPhase implements the all-gather phase of ring-allreduce
// func (c *GPUCoordinator) allGatherPhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
// 	numDevices := len(comm.devices)
// 	chunkSize := req.Count / uint64(numDevices)

// 	for step := 0; step < numDevices-1; step++ {
// 		for rank := 0; rank < numDevices; rank++ {
// 			sendRank := rank
// 			recvRank := (rank + 1) % numDevices

// 			// Calculate memory offsets for this step
// 			sendOffset := uint64(((rank - step + numDevices) % numDevices)) * chunkSize
// 			recvOffset := uint64(((rank - step - 1 + numDevices) % numDevices)) * chunkSize

// 			// Begin send operation
// 			sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
// 				SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + sendOffset},
// 				NumBytes:    chunkSize,
// 				DstRank:    &pb.Rank{Value: uint32(recvRank)},
// 			})
// 			if err != nil {
// 				return fmt.Errorf("failed to begin send: %v", err)
// 			}

// 			// Begin receive operation
// 			_, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
// 				StreamId:     sendResp.StreamId,
// 				RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + recvOffset},
// 				NumBytes:    chunkSize,
// 				SrcRank:     &pb.Rank{Value: uint32(sendRank)},
// 			})
// 			if err != nil {
// 				return fmt.Errorf("failed to begin receive: %v", err)
// 			}

// 			// Wait for completion
// 			for {
// 				statusResp, err := comm.devices[recvRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
// 					StreamId: sendResp.StreamId,
// 				})
// 				if err != nil {
// 					return fmt.Errorf("failed to get stream status: %v", err)
// 				}
// 				if statusResp.Status != pb.Status_IN_PROGRESS {
// 					break
// 				}
// 			}
// 		}
// 	}
// 	return nil
// }

// func (c *GPUCoordinator) allGatherPhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
//     numDevices := len(comm.devices)
//     chunkSize := req.Count / uint64(numDevices)

//     for step := 0; step < numDevices-1; step++ {
//         for rank := 0; rank < numDevices; rank++ {
//             sendRank := rank
//             recvRank := (rank + 1) % numDevices

//             // Calculate memory offsets for this step
//             sendOffset := uint64(((rank - step + numDevices) % numDevices)) * chunkSize
//             recvOffset := uint64(((rank - step - 1 + numDevices) % numDevices)) * chunkSize

//             // Begin send operation
//             sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
//                 SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + sendOffset},
//                 NumBytes:    chunkSize,
//                 DstRank:    &pb.Rank{Value: uint32(recvRank)},
//             })
//             if err != nil {
//                 return fmt.Errorf("failed to begin send: %v", err)
//             }

//             // Add a small delay to ensure the stream is registered
//             time.Sleep(10 * time.Millisecond)

//             // Begin receive operation
//             _, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
//                 StreamId:     sendResp.StreamId,
//                 RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + recvOffset},
//                 NumBytes:    chunkSize,
//                 SrcRank:     &pb.Rank{Value: uint32(sendRank)},
//             })
//             if err != nil {
//                 return fmt.Errorf("failed to begin receive: %v", err)
//             }

//             // Wait for completion
//             for {
//                 statusResp, err := comm.devices[recvRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
//                     StreamId: sendResp.StreamId,
//                 })
//                 if err != nil {
//                     return fmt.Errorf("failed to get stream status: %v", err)
//                 }
//                 if statusResp.Status != pb.Status_IN_PROGRESS {
//                     break
//                 }
//                 time.Sleep(10 * time.Millisecond)
//             }
//         }
//     }
//     return nil
// }

func (c *GPUCoordinator) allGatherPhase2(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
    numDevices := len(comm.devices)
    chunkSize := req.Count / uint64(numDevices)
    
    log.Printf("Starting all-gather phase with %d devices", numDevices)
    
    // All-gather phase
    for step := 0; step < numDevices-1; step++ {
        log.Printf("All-gather step %d/%d", step+1, numDevices-1)
        
        var wg sync.WaitGroup
        errChan := make(chan error, numDevices)
        
        for rank := 0; rank < numDevices; rank++ {
            wg.Add(1)
            go func(rank int) {
                defer wg.Done()
                
                sendRank := rank
                recvRank := (rank + 1) % numDevices
                
                // Calculate offset based on which chunk to propagate
                sendChunk := (rank - step - 1 + numDevices) % numDevices
                recvChunk := (rank - step - 2 + numDevices) % numDevices
                
                sendOffset := uint64(sendChunk) * chunkSize
                recvOffset := uint64(recvChunk) * chunkSize
                
                log.Printf("Rank %d sending chunk %d (offset %d) to rank %d, receiving at chunk %d (offset %d)",
                    sendRank, sendChunk, sendOffset, recvRank, recvChunk, recvOffset)
                
                // Begin send
                sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
                    SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + sendOffset},
                    NumBytes:    chunkSize,
                    DstRank:    &pb.Rank{Value: uint32(recvRank)},
                })
                if err != nil {
                    errChan <- fmt.Errorf("send failed at step %d, rank %d: %v", step, sendRank, err)
                    return
                }
                
                // Begin receive
                _, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
                    StreamId:     sendResp.StreamId,
                    RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + recvOffset},
                    NumBytes:    chunkSize,
                    SrcRank:     &pb.Rank{Value: uint32(sendRank)},
                })
                if err != nil {
                    errChan <- fmt.Errorf("receive failed at step %d, rank %d: %v", step, recvRank, err)
                    return
                }
                
                // Wait for completion
                for {
                    statusResp, err := comm.devices[recvRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
                        StreamId: sendResp.StreamId,
                    })
                    if err != nil {
                        errChan <- fmt.Errorf("status check failed: %v", err)
                        return
                    }
                    if statusResp.Status != pb.Status_IN_PROGRESS {
                        if statusResp.Status == pb.Status_FAILED {
                            errChan <- fmt.Errorf("operation failed at step %d between ranks %d->%d", 
                                step, sendRank, recvRank)
                            return
                        }
                        break
                    }
                    time.Sleep(1 * time.Millisecond)
                }
                
                log.Printf("Completed transfer step %d: rank %d -> rank %d", step, sendRank, recvRank)
            }(rank)
        }
        
        // Wait for all transfers in this step
        go func() {
            wg.Wait()
            close(errChan)
        }()
        
        // Check for errors
        for err := range errChan {
            if err != nil {
                return fmt.Errorf("all-gather failed: %v", err)
            }
        }
        
        // Small delay between steps to ensure stability
        time.Sleep(10 * time.Millisecond)
    }
    
    log.Printf("All-gather phase completed successfully")
    return nil
}

func (c *GPUCoordinator) allGatherPhase3(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
    numDevices := len(comm.devices)
    chunkSize := req.Count / uint64(numDevices)
    
    log.Printf("Starting all-gather phase with %d devices")
    
    // In all-gather phase, each device has one final chunk
    // Need to distribute these chunks to all devices
    for step := 0; step < numDevices-1; step++ {
        for rank := 0; rank < numDevices; rank++ {
            sendRank := rank
            recvRank := (rank + 1) % numDevices
            
            // Calculate which chunk to propagate
            // In step 0: device i sends chunk i-1
            // In step 1: device i sends chunk i-2
            // etc.
            chunkIndex := (rank - step - 1 + numDevices) % numDevices
            offset := uint64(chunkIndex) * chunkSize
            
            log.Printf("Step %d: Device %d sending chunk %d to device %d (offset=%d)", 
                step, sendRank, chunkIndex, recvRank, offset)
            
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
    }
    
    return nil
}

func (c *GPUCoordinator) allGatherPhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
    numDevices := len(comm.devices)
    chunkSize := req.Count / uint64(numDevices)
    
    log.Printf("Starting all-gather phase with %d devices", numDevices)
    
    // After scatter-reduce, for device i:
    // - The final reduced chunk (i-1) is at offset (i-1)*chunkSize
    for step := 0; step < numDevices-1; step++ {
        log.Printf("All-gather step %d/%d", step+1, numDevices-1)
        
        for rank := 0; rank < numDevices; rank++ {
            sendRank := rank
            recvRank := (rank + 1) % numDevices
            
            // Calculate which chunk this device has fully reduced
            // Device i has the final reduced value for chunk (i-1)
            reducedChunkIndex := (rank - 1 + numDevices) % numDevices
            offset := uint64(reducedChunkIndex) * chunkSize
            
            log.Printf("Step %d: Device %d sending its reduced chunk %d to device %d at offset %d", 
                step, sendRank, reducedChunkIndex, recvRank, offset)
            
            // Begin send from where the reduced value is
            sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
                SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + offset},
                NumBytes:    chunkSize,
                DstRank:    &pb.Rank{Value: uint32(recvRank)},
            })
            if err != nil {
                return fmt.Errorf("all-gather send failed at step %d, rank %d: %v", step, sendRank, err)
            }
            
            // Write to the same chunk position in destination
            _, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
                StreamId:     sendResp.StreamId,
                RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + offset},
                NumBytes:    chunkSize,
                SrcRank:     &pb.Rank{Value: uint32(sendRank)},
            })
            if err != nil {
                return fmt.Errorf("all-gather receive failed at step %d, rank %d: %v", step, recvRank, err)
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
        
        time.Sleep(5 * time.Millisecond)
    }
    
    return nil
}
// Memcpy handles memory copy operations between host and device
// func (c *GPUCoordinator) Memcpy(ctx context.Context, req *pb.MemcpyRequest) (*pb.MemcpyResponse, error) {
// 	switch v := req.Either.(type) {
// 	case *pb.MemcpyRequest_HostToDevice:
// 		return &pb.MemcpyResponse{
// 			Either: &pb.MemcpyResponse_HostToDevice{
// 				HostToDevice: &pb.MemcpyHostToDeviceResponse{
// 					Success: true,
// 				},
// 			},
// 		}, nil

// 	case *pb.MemcpyRequest_DeviceToHost:
// 		return &pb.MemcpyResponse{
// 			Either: &pb.MemcpyResponse_DeviceToHost{
// 				DeviceToHost: &pb.MemcpyDeviceToHostResponse{
// 					DstData: make([]byte, v.DeviceToHost.NumBytes),
// 				},
// 			},
// 		}, nil

// 	default:
// 		return nil, status.Error(codes.InvalidArgument, "invalid memcpy request type")
// 	}
// }

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

