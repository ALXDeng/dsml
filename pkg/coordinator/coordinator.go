// pkg/coordinator/coordinator.go
package coordinator

import (
	"context"
	pb "dsml/proto/gpu_sim"
	"fmt"
	"sync"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Communicator represents a group of GPU devices that can communicate with each other
type Communicator struct {
	id      uint64
	devices []pb.GPUDeviceClient
	status  pb.Status
	inGroup bool  // Whether a group operation is in progress
}

// GPUCoordinator manages multiple GPU devices and coordinates their operations
type GPUCoordinator struct {
	pb.UnimplementedGPUCoordinatorServer
	comms     map[uint64]*Communicator  // Map of communicator ID to Communicator
	commLock  sync.RWMutex
	nextCommID uint64
}

func NewGPUCoordinator() *GPUCoordinator {
	return &GPUCoordinator{
		comms:     make(map[uint64]*Communicator),
		nextCommID: 1,
	}
}

// CommInit initializes a new communicator with the specified number of devices
func (c *GPUCoordinator) CommInit(ctx context.Context, req *pb.CommInitRequest) (*pb.CommInitResponse, error) {
	if req.NumDevices < 1 {
		return nil, status.Error(codes.InvalidArgument, "number of devices must be positive")
	}

	// In a real implementation, we would discover and connect to actual GPU devices
	devices := make([]pb.GPUDeviceClient, req.NumDevices)
	deviceMetas := make([]*pb.DeviceMetadata, req.NumDevices)

	// For simulation, create metadata for each device
	for i := uint32(0); i < req.NumDevices; i++ {
		// In production, you would connect to real GPU devices here
		deviceMetas[i] = &pb.DeviceMetadata{
			DeviceId: &pb.DeviceId{Value: uint64(i)},
			MinMemAddr: &pb.MemAddr{Value: 0},
			MaxMemAddr: &pb.MemAddr{Value: 1024 * 1024}, // 1MB simulated memory
		}
	}

	// Create new communicator
	c.commLock.Lock()
	commId := c.nextCommID
	c.nextCommID++
	comm := &Communicator{
		id:      commId,
		devices: devices,
		status:  pb.Status_IN_PROGRESS,
		inGroup: false,
	}
	c.comms[commId] = comm
	c.commLock.Unlock()

	return &pb.CommInitResponse{
		Success: true,
		CommId:  commId,
		Devices: deviceMetas,
	}, nil
}

// GetCommStatus returns the status of a communicator
func (c *GPUCoordinator) GetCommStatus(ctx context.Context, req *pb.GetCommStatusRequest) (*pb.GetCommStatusResponse, error) {
	c.commLock.RLock()
	comm, exists := c.comms[req.CommId]
	c.commLock.RUnlock()

	if !exists {
		return nil, status.Error(codes.NotFound, "communicator not found")
	}

	return &pb.GetCommStatusResponse{
		Status: comm.status,
	}, nil
}

// GroupStart begins a group operation
func (c *GPUCoordinator) GroupStart(ctx context.Context, req *pb.GroupStartRequest) (*pb.GroupStartResponse, error) {
	c.commLock.Lock()
	defer c.commLock.Unlock()

	comm, exists := c.comms[req.CommId]
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
	c.commLock.Lock()
	defer c.commLock.Unlock()

	comm, exists := c.comms[req.CommId]
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
	c.commLock.RLock()
	comm, exists := c.comms[req.CommId]
	c.commLock.RUnlock()

	if !exists {
		return nil, status.Error(codes.NotFound, "communicator not found")
	}

	numDevices := len(comm.devices)
	if numDevices < 2 {
		return nil, status.Error(codes.FailedPrecondition, "need at least 2 devices for ring all-reduce")
	}

	// Phase 1: Ring-based scatter-reduce
	if err := c.scatterReducePhase(ctx, comm, req); err != nil {
		return nil, err
	}

	// Phase 2: Ring-based all-gather
	if err := c.allGatherPhase(ctx, comm, req); err != nil {
		return nil, err
	}

	return &pb.AllReduceRingResponse{Success: true}, nil
}

// scatterReducePhase implements the scatter-reduce phase of ring-allreduce
func (c *GPUCoordinator) scatterReducePhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
	numDevices := len(comm.devices)
	chunkSize := req.Count / uint64(numDevices)

	for step := 0; step < numDevices-1; step++ {
		for rank := 0; rank < numDevices; rank++ {
			sendRank := rank
			recvRank := (rank + 1) % numDevices

			// Calculate memory offsets for this step
			sendOffset := uint64(((rank - step + numDevices) % numDevices)) * chunkSize
			recvOffset := uint64(((rank - step - 1 + numDevices) % numDevices)) * chunkSize

			// Begin send operation
			sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
				SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + sendOffset},
				NumBytes:    chunkSize,
				DstRank:    &pb.Rank{Value: uint32(recvRank)},
			})
			if err != nil {
				return fmt.Errorf("failed to begin send: %v", err)
			}

			// Begin receive operation
			_, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
				StreamId:     sendResp.StreamId,
				RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + recvOffset},
				NumBytes:    chunkSize,
				SrcRank:     &pb.Rank{Value: uint32(sendRank)},
			})
			if err != nil {
				return fmt.Errorf("failed to begin receive: %v", err)
			}

			// Wait for completion
			for {
				statusResp, err := comm.devices[recvRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
					StreamId: sendResp.StreamId,
				})
				if err != nil {
					return fmt.Errorf("failed to get stream status: %v", err)
				}
				if statusResp.Status != pb.Status_IN_PROGRESS {
					break
				}
			}
		}
	}
	return nil
}

// allGatherPhase implements the all-gather phase of ring-allreduce
func (c *GPUCoordinator) allGatherPhase(ctx context.Context, comm *Communicator, req *pb.AllReduceRingRequest) error {
	numDevices := len(comm.devices)
	chunkSize := req.Count / uint64(numDevices)

	for step := 0; step < numDevices-1; step++ {
		for rank := 0; rank < numDevices; rank++ {
			sendRank := rank
			recvRank := (rank + 1) % numDevices

			// Calculate memory offsets for this step
			sendOffset := uint64(((rank - step + numDevices) % numDevices)) * chunkSize
			recvOffset := uint64(((rank - step - 1 + numDevices) % numDevices)) * chunkSize

			// Begin send operation
			sendResp, err := comm.devices[sendRank].BeginSend(ctx, &pb.BeginSendRequest{
				SendBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(sendRank)].Value + sendOffset},
				NumBytes:    chunkSize,
				DstRank:    &pb.Rank{Value: uint32(recvRank)},
			})
			if err != nil {
				return fmt.Errorf("failed to begin send: %v", err)
			}

			// Begin receive operation
			_, err = comm.devices[recvRank].BeginReceive(ctx, &pb.BeginReceiveRequest{
				StreamId:     sendResp.StreamId,
				RecvBuffAddr: &pb.MemAddr{Value: req.MemAddrs[uint32(recvRank)].Value + recvOffset},
				NumBytes:    chunkSize,
				SrcRank:     &pb.Rank{Value: uint32(sendRank)},
			})
			if err != nil {
				return fmt.Errorf("failed to begin receive: %v", err)
			}

			// Wait for completion
			for {
				statusResp, err := comm.devices[recvRank].GetStreamStatus(ctx, &pb.GetStreamStatusRequest{
					StreamId: sendResp.StreamId,
				})
				if err != nil {
					return fmt.Errorf("failed to get stream status: %v", err)
				}
				if statusResp.Status != pb.Status_IN_PROGRESS {
					break
				}
			}
		}
	}
	return nil
}

// Memcpy handles memory copy operations between host and device
func (c *GPUCoordinator) Memcpy(ctx context.Context, req *pb.MemcpyRequest) (*pb.MemcpyResponse, error) {
	switch v := req.Either.(type) {
	case *pb.MemcpyRequest_HostToDevice:
		return &pb.MemcpyResponse{
			Either: &pb.MemcpyResponse_HostToDevice{
				HostToDevice: &pb.MemcpyHostToDeviceResponse{
					Success: true,
				},
			},
		}, nil

	case *pb.MemcpyRequest_DeviceToHost:
		return &pb.MemcpyResponse{
			Either: &pb.MemcpyResponse_DeviceToHost{
				DeviceToHost: &pb.MemcpyDeviceToHostResponse{
					DstData: make([]byte, v.DeviceToHost.NumBytes),
				},
			},
		}, nil

	default:
		return nil, status.Error(codes.InvalidArgument, "invalid memcpy request type")
	}
}