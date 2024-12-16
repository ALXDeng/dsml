// cmd/device/main.go
package main

import (
	"dsml/pkg/device"
	"flag"
	"fmt"
	"log"
	"net"

	pb "github.com/ALXDeng/dsml/proto/gpu_sim"

	"google.golang.org/grpc"
)

func main() {
	port := flag.Int("port", 50051, "The server port")
	memorySize := flag.Uint64("memory", 1024*1024*1024, "GPU memory size in bytes")
	flag.Parse()

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	gpuDevice := device.NewGPUDevice(*memorySize)
	grpcServer := grpc.NewServer()
	pb.RegisterGPUDeviceServer(grpcServer, gpuDevice)

	log.Printf("GPU Device server listening at %v", lis.Addr())
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}