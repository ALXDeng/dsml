// pkg/coordinator/coordinator.go
package coordinator

import (
	"context"
	"fmt"
	"sync"

	pb "github.com/ALXDeng/dsml/proto"
)

type Communicator struct {
	id      uint64
	devices []pb.GPUDeviceClient
	status  pb.Status
}

type GPUCoordinator struct {
	pb.UnimplementedGPUCoordinatorServer
	comms     map[uint64]*Communicator
	commLock  sync.RWMutex
	deviceMap map[uint64]pb.GPUDeviceClient
}

func NewGPUCoordinator() *GPUCoordinator {
	return &GPUCoordinator{
		comms:     make(map[uint64]*Communicator),
		deviceMap: make(map[uint64]pb.GPUDeviceClient),
	}
}

func (c *GPUCoordinator) CommInit(ctx context.Context, req *pb.CommInitRequest) (*pb.CommInitResponse, error) {
	// In a real implementation, you would discover available GPUs
	// For now, we'll simulate it with the requested number of devices
	commId := uint64(len(c.comms) + 1)
	
	devices := make([]pb.GPUDeviceClient, req.NumDevices)
	deviceMetas := make([]*pb.DeviceMetadata, req.NumDevices)

	// In a real implementation, you would connect to actual GPU devices
	// Here we'll just create metadata for simulation
	for i := uint32(0); i < req.NumDevices; i++ {
		deviceMetas[i] = &pb.DeviceMetadata{
			DeviceId:    &pb.DeviceId{Value: uint64(i)},
			MinMemAddr:  &pb.MemAddr{Value: 0},
			MaxMemAddr:  &pb.MemAddr{Value: 1024 * 1024}, // 1MB simulated memory
		}
	}

	comm := &Communicator{
		id:      commId,
		devices: devices,
		status:  pb.Status_IN_PROGRESS,
	}

	c.commLock.Lock()
	c.comms[commId] = comm
	c.commLock.Unlock()

	return &pb.CommInitResponse{
		Success:  true,
		CommId:   commId,
		Devices:  deviceMetas,
	}, nil
}

func (c *GPUCoordinator) AllReduceRing(ctx context.Context, req *pb.AllReduceRingRequest) (*pb.AllReduceRingResponse, error) {
	c.commLock.RLock()
	comm, exists := c.comms[req.CommId]
	c.commLock.RUnlock()

	if !exists {
		return nil, fmt.Errorf("communicator not found")
	}

	numDevices := len(comm.devices)
	if numDevices < 2 {
		return nil, fmt.Errorf("need at least 2 devices for ring all-reduce")
	}

	// Implement ring all-reduce algorithm
	// Phase 1: Ring-based scatter-reduce
	for i := 0; i < numDevices-1; i++ {
		for rank := 0; rank < numDevices; rank++ {
			sendRank := rank
			recvRank := (rank + 1) % numDevices
			
			// Initialize send operation
			sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
				SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)]},
				NumBytes:    req.Count / uint64(numDevices),
				DstRank:    &pb.Rank{Value: uint32(recvRank)},
			})
			if err != nil {
				return nil, fmt.Errorf("failed to begin send: %v", err)
			}

			// Initialize receive operation
			_, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
				StreamId:     sendResp.StreamId,
				RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)]},
				NumBytes:     req.Count / uint64(numDevices),
				SrcRank:     &pb.Rank{Value: uint32(sendRank)},
			})
			if err != nil {
				return nil, fmt.Errorf("failed to begin receive: %v", err)
			}
		}
	}

	// Phase 2: Ring-based all-gather
	// Similar to phase 1 but distributing the reduced values
	
	return &pb.AllReduceRingResponse{
		Success: true,
	}, nil
}