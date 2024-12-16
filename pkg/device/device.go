package device

import (
	"context"
	"encoding/binary"
	"io"
	"math"
	"math/rand"
	"sync"

	pb "dsml/proto/gpu_sim"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var (
    globalStreams = make(map[uint64]*Stream)
    globalStreamsMutex sync.RWMutex
)

type GPUDevice struct {
    pb.UnimplementedGPUDeviceServer
    deviceId    uint64
    memory     []byte
    minAddr    uint64
    maxAddr    uint64
    rank       uint32     // Add rank for device identification
}

type Stream struct {
    status     pb.Status
    srcRank    uint32
    dstRank    uint32
    srcAddr    uint64
    dstAddr    uint64
    numBytes   uint64
    data       []byte
    reducedData []byte  // Add this to store reduced data
}
func NewGPUDevice(memorySize uint64, rank uint32) *GPUDevice {
    return &GPUDevice{
        deviceId: rand.Uint64(),
        memory:   make([]byte, memorySize),
        minAddr:  0,
        maxAddr:  memorySize,
        rank:     rank,
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
        srcRank:  d.rank,
        dstRank:  req.DstRank.Value,
        srcAddr:  req.SendBuffAddr.Value,
        numBytes: req.NumBytes,
        data:     make([]byte, req.NumBytes),
    }
    
    // Copy data from device memory to stream
    copy(stream.data, d.memory[stream.srcAddr:stream.srcAddr+stream.numBytes])

    globalStreamsMutex.Lock()
    globalStreams[streamId] = stream
    globalStreamsMutex.Unlock()

    return &pb.BeginSendResponse{
        Initiated: true,
        StreamId:  &pb.StreamId{Value: streamId},
    }, nil
}

func (d *GPUDevice) BeginReceive(ctx context.Context, req *pb.BeginReceiveRequest) (*pb.BeginReceiveResponse, error) {
    globalStreamsMutex.RLock()
    stream, exists := globalStreams[req.StreamId.Value]
    globalStreamsMutex.RUnlock()

    if !exists {
        return nil, status.Errorf(codes.NotFound, "stream not found")
    }

    // Convert bytes to float64s for both incoming and existing data
    numFloats := stream.numBytes / 8
    incomingData := make([]float64, numFloats)
    existingData := make([]float64, numFloats)

    // Convert incoming stream data to float64s
    for i := 0; i < int(numFloats); i++ {
        incomingData[i] = math.Float64frombits(binary.LittleEndian.Uint64(stream.data[i*8 : (i+1)*8]))
    }

    // Get existing data from device memory
    for i := 0; i < int(numFloats); i++ {
        existingData[i] = math.Float64frombits(binary.LittleEndian.Uint64(d.memory[req.RecvBuffAddr.Value+uint64(i*8) : req.RecvBuffAddr.Value+uint64((i+1)*8)]))
    }

    // Perform reduction (SUM)
    resultData := make([]byte, stream.numBytes)
    for i := 0; i < int(numFloats); i++ {
        sum := incomingData[i] + existingData[i]
        binary.LittleEndian.PutUint64(resultData[i*8:], math.Float64bits(sum))
    }

    // Store the result in device memory
    copy(d.memory[req.RecvBuffAddr.Value:], resultData)

    // Update stream with reduced data
    stream.data = resultData
    stream.status = pb.Status_SUCCESS

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
    globalStreamsMutex.RLock()
    stream, exists := globalStreams[req.StreamId.Value]
    globalStreamsMutex.RUnlock()

    if !exists {
        return nil, status.Errorf(codes.NotFound, "stream not found")
    }

    return &pb.GetStreamStatusResponse{
        Status: stream.status,
    }, nil
}