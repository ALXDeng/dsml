// cmd/coordinator/main.go
package main

import (
	"flag"
	"fmt"
	"log"
	"net"

	"dsml/pkg/coordinator"
	pb "dsml/proto/gpu_sim"

	"google.golang.org/grpc"
)

func main() {
	port := flag.Int("port", 50052, "The server port")
	flag.Parse()

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	gpuCoordinator := coordinator.NewGPUCoordinator()
	grpcServer := grpc.NewServer()
	pb.RegisterGPUCoordinatorServer(grpcServer, gpuCoordinator)

	log.Printf("GPU Coordinator server listening at %v", lis.Addr())
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}