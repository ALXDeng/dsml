// pck/device/device.go
package device

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sync"
	"time"

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
//     Rank       uint32     
// }

type GPUDevice1 struct {
    pb.UnimplementedGPUDeviceServer
    DeviceId uint64
    Memory   []byte
    MinAddr  uint64
    MaxAddr  uint64
    Rank     uint32
    memLock  sync.RWMutex
    IsAllGather bool 
}
const (
    HealthCheckTimeout = 2 * time.Second  // Maximum time without activity before considered unhealthy
    HistorySize       = 10               // Size of circular buffer for request history
    FailureThreshold  = 3                // Number of failures needed to mark device as failed
)

type GPUDevice struct {
    pb.UnimplementedGPUDeviceServer
    DeviceId    uint64
    Memory      []byte
    MinAddr     uint64
    MaxAddr     uint64
    Rank        uint32
    MemLock     sync.RWMutex
    IsAllGather bool
    
    // Failure detection fields
    lastActive     time.Time
    requestHistory []bool      // Circular buffer of recent request successes
    historyIndex   int        // Current index in circular buffer
    healthLock     sync.Mutex // Protects failure detection fields
}

func (d *GPUDevice) SetAllGatherPhase(enabled bool) {
    d.MemLock.Lock()
    d.IsAllGather = enabled
    // log.Printf("Device %d phase changed: isAllGather=%v", d.Rank, enabled)
    d.MemLock.Unlock()
}
func (d *GPUDevice) recordActivity(success bool) {
    d.healthLock.Lock()
    defer d.healthLock.Unlock()
    
    d.lastActive = time.Now()
    d.requestHistory[d.historyIndex] = success
    d.historyIndex = (d.historyIndex + 1) % HistorySize
}

func (d *GPUDevice) IsHealthy() bool {
    d.healthLock.Lock()
    defer d.healthLock.Unlock()
    
    // Check if device has been inactive too long
    if time.Since(d.lastActive) > HealthCheckTimeout {
        return false
    }
    
    // Count recent failures
    failures := 0
    for _, success := range d.requestHistory {
        if !success {
            failures++
        }
    }
    
    return failures < FailureThreshold
}

type Stream struct {
    status      pb.Status
    srcRank     uint32
    dstRank     uint32
    srcAddr     uint64
    dstAddr     uint64
    numBytes    uint64
    data        []byte
    inAllGather bool  
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
        DeviceId:       uint64(rank),
        Memory:         memory,
        MinAddr:        0,
        MaxAddr:        memorySize,
        Rank:          rank,
        MemLock:       sync.RWMutex{},
        IsAllGather:   false,
        // Initialize failure detection fields
        lastActive:     time.Now(),
        requestHistory: make([]bool, HistorySize),  // Initialize the circular buffer
        historyIndex:  0,
        healthLock:    sync.Mutex{},
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


func (d *GPUDevice) BeginSend(ctx context.Context, req *pb.BeginSendRequest) (*pb.BeginSendResponse, error) {
    streamId := rand.Uint64()

	success := true
    defer d.recordActivity(success)
    
    if req.SendBuffAddr.Value+req.NumBytes > d.MaxAddr {
        return nil, fmt.Errorf("send buffer address out of bounds: addr=%d, size=%d, max=%d",
            req.SendBuffAddr.Value, req.NumBytes, d.MaxAddr)
    }
    
    
    stream := &Stream{
        status:   pb.Status_IN_PROGRESS,
        srcRank:  d.Rank,
        dstRank:  req.DstRank.Value,
        srcAddr:  req.SendBuffAddr.Value,
        numBytes: req.NumBytes,
        data:     make([]byte, req.NumBytes),
    }
    
    d.MemLock.RLock()
    copy(stream.data, d.Memory[stream.srcAddr:stream.srcAddr+stream.numBytes])
    d.MemLock.RUnlock()
    

    values := make([]float64, req.NumBytes/8)
    for i := range values {
        values[i] = math.Float64frombits(binary.LittleEndian.Uint64(stream.data[i*8:]))
    }
    // log.Printf("Device %d: Sending chunk - addr=%d, values=%v", 
        // d.Rank, req.SendBuffAddr.Value, values)
    
    globalStreamsMutex.Lock()
    globalStreams[streamId] = stream
    globalStreamsMutex.Unlock()
    
    return &pb.BeginSendResponse{
        Initiated: true,
        StreamId:  &pb.StreamId{Value: streamId},
    }, nil
}


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

    
    incomingVals := make([]float64, req.NumBytes/8)
    for i := range incomingVals {
        incomingVals[i] = math.Float64frombits(binary.LittleEndian.Uint64(stream.data[i*8:]))
    }

    d.MemLock.RLock()
    existingVals := make([]float64, req.NumBytes/8)
    for i := range existingVals {
        existingVals[i] = math.Float64frombits(binary.LittleEndian.Uint64(
            d.Memory[req.RecvBuffAddr.Value+uint64(i*8):]))
    }
    isAllGather := d.IsAllGather
    d.MemLock.RUnlock()

    // log.Printf("Device %d: Processing at addr=%d, isAllGather=%v incoming=%v existing=%v", 
        // d.Rank, req.RecvBuffAddr.Value, isAllGather, incomingVals, existingVals)

    resultVals := make([]float64, len(incomingVals))
    if !isAllGather {
        // Scatter-reduce phase: sum values
        for i := range resultVals {
            resultVals[i] = incomingVals[i] + existingVals[i]
        }
    } else {
        copy(resultVals, incomingVals)
    }

    
    resultData := make([]byte, req.NumBytes)
    for i := range resultVals {
        binary.LittleEndian.PutUint64(resultData[i*8:], math.Float64bits(resultVals[i]))
    }

    d.MemLock.Lock()
    copy(d.Memory[req.RecvBuffAddr.Value:], resultData)
    d.MemLock.Unlock()

    // log.Printf("Device %d: Result values=%v at addr=%d", 
        // d.Rank, resultVals, req.RecvBuffAddr.Value)

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


func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
func (d *GPUDevice) SetLastActiveTime(t time.Time) {
    d.healthLock.Lock()
    defer d.healthLock.Unlock()
    d.lastActive = t
}

func (d *GPUDevice) IsDeviceHealthy() bool {
    d.healthLock.Lock()
    defer d.healthLock.Unlock()
    return time.Since(d.lastActive) <= HealthCheckTimeout
}
