// pck/device/device.go
package device

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log"
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

// type GPUDevice struct {
//     pb.UnimplementedGPUDeviceServer
//     DeviceId    uint64
//     Memory     []byte
//     MinAddr    uint64
//     MaxAddr    uint64
//     Rank       uint32     // Add rank for device identification
// }

type GPUDevice struct {
    pb.UnimplementedGPUDeviceServer
    DeviceId uint64
    Memory   []byte
    MinAddr  uint64
    MaxAddr  uint64
    Rank     uint32
    memLock  sync.RWMutex
    IsAllGather bool  // New flag to track which phase we're in
}
func (d *GPUDevice) SetAllGatherPhase(enabled bool) {
    d.memLock.Lock()
    d.IsAllGather = enabled
    log.Printf("Device %d phase changed: isAllGather=%v", d.Rank, enabled)
    d.memLock.Unlock()
}

type Stream struct {
    status      pb.Status
    srcRank     uint32
    dstRank     uint32
    srcAddr     uint64
    dstAddr     uint64
    numBytes    uint64
    data        []byte
    inAllGather bool  // New flag to distinguish phases
}

// func NewGPUDevice(memorySize uint64, rank uint32) *GPUDevice {
//     return &GPUDevice{
//         DeviceId: uint64(rank),
//         Memory:   make([]byte, memorySize),
//         MinAddr:  0,
//         MaxAddr:  memorySize,
//         Rank:     rank,
//     }
// }
func NewGPUDevice(memorySize uint64, rank uint32) *GPUDevice {
    memory := make([]byte, memorySize)
    
    // Initialize memory to zeros
    for i := range memory {
        memory[i] = 0
    }
    
    return &GPUDevice{
        DeviceId: uint64(rank),
        Memory:   memory,
        MinAddr:  0,
        MaxAddr:  memorySize,
        Rank:     rank,
        memLock:  sync.RWMutex{},
		IsAllGather: false,
    }
}


func (d *GPUDevice) GetDeviceMetadata(ctx context.Context, req *pb.GetDeviceMetadataRequest) (*pb.GetDeviceMetadataResponse, error) {
	return &pb.GetDeviceMetadataResponse{
		Metadata: &pb.DeviceMetadata{
			DeviceId:    &pb.DeviceId{Value: d.DeviceId},
			MinMemAddr:  &pb.MemAddr{Value: d.MinAddr},
			MaxMemAddr:  &pb.MemAddr{Value: d.MaxAddr},
		},
	}, nil
}
func (d *GPUDevice) BeginSend1(ctx context.Context, req *pb.BeginSendRequest) (*pb.BeginSendResponse, error) {
    streamId := rand.Uint64()
    
    stream := &Stream{
        status:   pb.Status_IN_PROGRESS,
        srcRank:  d.Rank,
        dstRank:  req.DstRank.Value,
        srcAddr:  req.SendBuffAddr.Value,
        numBytes: req.NumBytes,
        data:     make([]byte, req.NumBytes),
    }
    
    // Copy data from device memory to stream
    copy(stream.data, d.Memory[stream.srcAddr:stream.srcAddr+stream.numBytes])

    globalStreamsMutex.Lock()
    globalStreams[streamId] = stream
    globalStreamsMutex.Unlock()

    return &pb.BeginSendResponse{
        Initiated: true,
        StreamId:  &pb.StreamId{Value: streamId},
    }, nil
}

func (d *GPUDevice) BeginSend(ctx context.Context, req *pb.BeginSendRequest) (*pb.BeginSendResponse, error) {
    streamId := rand.Uint64()
    
    if req.SendBuffAddr.Value+req.NumBytes > d.MaxAddr {
        return nil, fmt.Errorf("send buffer address out of bounds: addr=%d, size=%d, max=%d",
            req.SendBuffAddr.Value, req.NumBytes, d.MaxAddr)
    }
    
    // Create stream with properly sized data buffer
    stream := &Stream{
        status:   pb.Status_IN_PROGRESS,
        srcRank:  d.Rank,
        dstRank:  req.DstRank.Value,
        srcAddr:  req.SendBuffAddr.Value,
        numBytes: req.NumBytes,
        data:     make([]byte, req.NumBytes),
    }
    
    // Safely copy data from device memory
    d.memLock.RLock()
    copy(stream.data, d.Memory[stream.srcAddr:stream.srcAddr+stream.numBytes])
    d.memLock.RUnlock()
    
    // Debug: Print the actual float64 values being sent
    values := make([]float64, req.NumBytes/8)
    for i := range values {
        values[i] = math.Float64frombits(binary.LittleEndian.Uint64(stream.data[i*8:]))
    }
    log.Printf("Device %d: Sending chunk - addr=%d, values=%v", 
        d.Rank, req.SendBuffAddr.Value, values)
    
    globalStreamsMutex.Lock()
    globalStreams[streamId] = stream
    globalStreamsMutex.Unlock()
    
    return &pb.BeginSendResponse{
        Initiated: true,
        StreamId:  &pb.StreamId{Value: streamId},
    }, nil
}



// func (d *GPUDevice) BeginReceive(ctx context.Context, req *pb.BeginReceiveRequest) (*pb.BeginReceiveResponse, error) {
//     if req.RecvBuffAddr.Value+req.NumBytes > d.MaxAddr {
//         return nil, fmt.Errorf("receive buffer address out of bounds")
//     }

//     globalStreamsMutex.RLock()
//     stream, exists := globalStreams[req.StreamId.Value]
//     globalStreamsMutex.RUnlock()

//     if !exists {
//         return nil, fmt.Errorf("stream not found")
//     }

//     // Read incoming data
//     incomingVals := make([]float64, req.NumBytes/8)
//     for i := range incomingVals {
//         incomingVals[i] = math.Float64frombits(binary.LittleEndian.Uint64(stream.data[i*8:]))
//     }

//     // Read existing data
//     d.memLock.RLock()
//     existingVals := make([]float64, req.NumBytes/8)
//     for i := range existingVals {
//         existingVals[i] = math.Float64frombits(binary.LittleEndian.Uint64(
//             d.Memory[req.RecvBuffAddr.Value+uint64(i*8):]))
//     }
//     isAllGather := d.IsAllGather
//     d.memLock.RUnlock()

//     log.Printf("Device %d: Processing at addr=%d, isAllGather=%v incoming=%v existing=%v", 
//         d.Rank, req.RecvBuffAddr.Value, isAllGather, incomingVals, existingVals)

//     // Prepare result buffer
//     resultVals := make([]float64, len(incomingVals))
//     resultData := make([]byte, req.NumBytes)

//     if !isAllGather {
//         // Scatter-reduce phase: perform reduction
//         for i := range resultVals {
//             resultVals[i] = incomingVals[i] + existingVals[i]
//         }
//     } else {
//         // All-gather phase: copy incoming values
//         // In all-gather, incoming values are the final reduced values
//         // for this chunk, so we just take them as-is
//         copy(resultVals, incomingVals)
//     }

//     // Convert back to bytes
//     for i := range resultVals {
//         binary.LittleEndian.PutUint64(resultData[i*8:], math.Float64bits(resultVals[i]))
//     }

//     // Update device memory
//     d.memLock.Lock()
//     copy(d.Memory[req.RecvBuffAddr.Value:], resultData)
//     d.memLock.Unlock()

//     log.Printf("Device %d: Result values=%v at addr=%d", 
//         d.Rank, resultVals, req.RecvBuffAddr.Value)

//     globalStreamsMutex.Lock()
//     stream.status = pb.Status_SUCCESS
//     globalStreamsMutex.Unlock()

//     return &pb.BeginReceiveResponse{
//         Initiated: true,
//     }, nil
// }

func (d *GPUDevice) BeginReceive(ctx context.Context, req *pb.BeginReceiveRequest) (*pb.BeginReceiveResponse, error) {
    if req.RecvBuffAddr.Value+req.NumBytes > d.MaxAddr {
        return nil, fmt.Errorf("receive buffer address out of bounds")
    }

    globalStreamsMutex.RLock()
    stream, exists := globalStreams[req.StreamId.Value]
    globalStreamsMutex.RUnlock()

    if !exists {
        return nil, fmt.Errorf("stream not found")
    }

    // Convert incoming data to float64s for clarity
    incomingVals := make([]float64, req.NumBytes/8)
    for i := range incomingVals {
        incomingVals[i] = math.Float64frombits(binary.LittleEndian.Uint64(stream.data[i*8:]))
    }

    // Read existing values
    d.memLock.RLock()
    existingVals := make([]float64, req.NumBytes/8)
    for i := range existingVals {
        existingVals[i] = math.Float64frombits(binary.LittleEndian.Uint64(
            d.Memory[req.RecvBuffAddr.Value+uint64(i*8):]))
    }
    isAllGather := d.IsAllGather
    d.memLock.RUnlock()

    log.Printf("Device %d: Processing at addr=%d, isAllGather=%v incoming=%v existing=%v", 
        d.Rank, req.RecvBuffAddr.Value, isAllGather, incomingVals, existingVals)

    // Process data based on phase
    resultVals := make([]float64, len(incomingVals))
    if !isAllGather {
        // Scatter-reduce phase: sum values
        for i := range resultVals {
            resultVals[i] = incomingVals[i] + existingVals[i]
        }
    } else {
        // All-gather phase: take incoming values
        // These are the final reduced sums for this chunk
        copy(resultVals, incomingVals)
    }

    // Convert results back to bytes
    resultData := make([]byte, req.NumBytes)
    for i := range resultVals {
        binary.LittleEndian.PutUint64(resultData[i*8:], math.Float64bits(resultVals[i]))
    }

    // Update device memory
    d.memLock.Lock()
    copy(d.Memory[req.RecvBuffAddr.Value:], resultData)
    d.memLock.Unlock()

    log.Printf("Device %d: Result values=%v at addr=%d", 
        d.Rank, resultVals, req.RecvBuffAddr.Value)

    globalStreamsMutex.Lock()
    stream.status = pb.Status_SUCCESS
    globalStreamsMutex.Unlock()

    return &pb.BeginReceiveResponse{
        Initiated: true,
    }, nil
}





func (d *GPUDevice) BeginReceive1(ctx context.Context, req *pb.BeginReceiveRequest) (*pb.BeginReceiveResponse, error) {
    globalStreamsMutex.RLock()
    stream, exists := globalStreams[req.StreamId.Value]
    globalStreamsMutex.RUnlock()
    
    if !exists {
        return nil, status.Errorf(codes.NotFound, "stream not found")
    }
    
    if req.RecvBuffAddr.Value+req.NumBytes > d.MaxAddr {
        return nil, status.Errorf(codes.InvalidArgument, "receive buffer address out of bounds")
    }

    // Convert incoming stream data to float64s
    numFloats := req.NumBytes / 8
    incomingData := make([]float64, numFloats)
    for i := 0; i < int(numFloats); i++ {
        bits := binary.LittleEndian.Uint64(stream.data[i*8:])
        val := math.Float64frombits(bits)
        incomingData[i] = val
    }

    // Get existing data
    existingData := make([]float64, numFloats)
    d.memLock.RLock()
    for i := 0; i < int(numFloats); i++ {
        addr := req.RecvBuffAddr.Value + uint64(i*8)
        bits := binary.LittleEndian.Uint64(d.Memory[addr : addr+8])
        existingData[i] = math.Float64frombits(bits)
    }
    d.memLock.RUnlock()

    // In all-gather phase, just copy the data without reduction
    resultData := make([]byte, req.NumBytes)
    if !d.IsAllGather {
        // We're in scatter-reduce phase, perform reduction
        log.Printf("Device %d: Reducing values at addr %d", d.Rank, req.RecvBuffAddr.Value)
        for i := 0; i < int(numFloats); i++ {
            result := incomingData[i] + existingData[i]
            binary.LittleEndian.PutUint64(resultData[i*8:], math.Float64bits(result))
        }
    } else {
        // We're in all-gather phase, just copy
        log.Printf("Device %d: Copying values at addr %d", d.Rank, req.RecvBuffAddr.Value)
        copy(resultData, stream.data)
    }

    // Update device memory
    d.memLock.Lock()
    copy(d.Memory[req.RecvBuffAddr.Value:], resultData)
    d.memLock.Unlock()

    log.Printf("Device %d: Values - incoming: %v, existing: %v, result: %v",
        d.Rank, incomingData, existingData, bytesToFloat64s(resultData))

    globalStreamsMutex.Lock()
    stream.status = pb.Status_SUCCESS
    globalStreamsMutex.Unlock()

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
func (d *GPUDevice) GetStreamStatus1(ctx context.Context, req *pb.GetStreamStatusRequest) (*pb.GetStreamStatusResponse, error) {
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

func (d *GPUDevice) DumpMemory(start, count uint64) []float64 {
    if start+count > d.MaxAddr {
        count = d.MaxAddr - start
    }
    
    result := make([]float64, count/8)
    for i := range result {
        result[i] = math.Float64frombits(binary.LittleEndian.Uint64(d.Memory[start+uint64(i*8):]))
    }
    return result
}



func bytesToFloat64s(data []byte) []float64 {
    result := make([]float64, len(data)/8)
    for i := range result {
        result[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
    }
    return result
}

// Helper function for min
func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}