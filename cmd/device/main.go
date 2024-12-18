package main

import (
	"dsml/pkg/device"
	"flag"
	"fmt"
	"log"
	"net"

	pb "dsml/proto/gpu_sim"

	"google.golang.org/grpc"
)

func main() {
	port := flag.Int("port", 50051, "The server port")
	memorySize := flag.Uint64("memory", 1024*1024*1024, "GPU memory size in bytes")
	rank := flag.Uint("rank", 0, "Device rank (default: 0)")
	flag.Parse()

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	gpuDevice := device.NewGPUDevice(*memorySize, uint32(*rank))
	grpcServer := grpc.NewServer()
	pb.RegisterGPUDeviceServer(grpcServer, gpuDevice)

	log.Printf("GPU Device server listening at %v (rank %d)", lis.Addr(), *rank)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
