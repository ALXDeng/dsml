package device

import (
	"context"
	pb "/proto/gpu_sim"
	"io"
	"math/rand"
	"sync"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type GPUDevice struct {
	pb.UnimplementedGPUDeviceServer
	deviceId    uint64
	memory     []byte
	minAddr    uint64
	maxAddr    uint64
	streams    map[uint64]*Stream
	streamLock sync.RWMutex
}

type Stream struct {
	status     pb.Status
	srcRank    uint32
	dstRank    uint32
	srcAddr    uint64
	dstAddr    uint64
	numBytes   uint64
	dataChan   chan []byte
}

func NewGPUDevice(memorySize uint64) *GPUDevice {
	deviceId := rand.Uint64()
	return &GPUDevice{
		deviceId: deviceId,
		memory:   make([]byte, memorySize),
		minAddr:  0,
		maxAddr:  memorySize,
		streams:  make(map[uint64]*Stream),
	}
}

func (d *GPUDevice) GetDeviceMetadata(ctx context.Context, req *pb.GetDeviceMetadataRequest) (*pb.GetDeviceMetadataResponse, error) {
	return &pb.GetDeviceMetadataResponse{
		Metadata: &pb.DeviceMetadata{
			DeviceId:    &pb.DeviceId{Value: d.deviceId},
			MinMemAddr:  &pb.MemAddr{Value: d.minAddr},
			MaxMemAddr:  &pb.MemAddr{Value: d.maxAddr},
		},
	}, nil
}

func (d *GPUDevice) BeginSend(ctx context.Context, req *pb.BeginSendRequest) (*pb.BeginSendResponse, error) {
	streamId := rand.Uint64()
	
	stream := &Stream{
		status:   pb.Status_IN_PROGRESS,
		srcAddr:  req.SendBuffAddr.Value,
		numBytes: req.NumBytes,
		dstRank:  req.DstRank.Value,
		dataChan: make(chan []byte, 1),
	}

	d.streamLock.Lock()
	d.streams[streamId] = stream
	d.streamLock.Unlock()

	return &pb.BeginSendResponse{
		Initiated: true,
		StreamId:  &pb.StreamId{Value: streamId},
	}, nil
}

func (d *GPUDevice) BeginReceive(ctx context.Context, req *pb.BeginReceiveRequest) (*pb.BeginReceiveResponse, error) {
	d.streamLock.Lock()
	stream, exists := d.streams[req.StreamId.Value]
	if !exists {
		d.streamLock.Unlock()
		return nil, status.Error(codes.NotFound, "stream not found")
	}
	
	stream.dstAddr = req.RecvBuffAddr.Value
	stream.srcRank = req.SrcRank.Value
	d.streamLock.Unlock()

	return &pb.BeginReceiveResponse{
		Initiated: true,
	}, nil
}

func (d *GPUDevice) StreamSend(stream pb.GPUDevice_StreamSendServer) error {
	var totalData []byte
	
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			// End of stream, process the complete data
			// In a real implementation, you would handle the data transfer here
			return stream.SendAndClose(&pb.StreamSendResponse{
				Success: true,
			})
		}
		if err != nil {
			return err
		}
		
		totalData = append(totalData, chunk.Data...)
	}
}

func (d *GPUDevice) GetStreamStatus(ctx context.Context, req *pb.GetStreamStatusRequest) (*pb.GetStreamStatusResponse, error) {
	d.streamLock.RLock()
	stream, exists := d.streams[req.StreamId.Value]
	d.streamLock.RUnlock()

	if !exists {
		return nil, status.Error(codes.NotFound, "stream not found")
	}

	return &pb.GetStreamStatusResponse{
		Status: stream.status,
	}, nil
}