[Sketch] Distributed Machine Learning Lab

# Overview
Modern advances in ML and AI rely heavily on training and running inference on
large models. Particularly, training models is often done on multi-GPU clusters and
training jobs can run for hours or even days.

Among challenges in orchestrating multi-GPU training jobs are high failure rates, complex low-level APIs, and lack of observability and debuggability. See S3.3.4 of [this paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) for more details.

This lab is designed to give you a taste of DSML and prompt you to build solutions in a simulated environment. As such, the protocols may appear awkward so as to closely mimic the NCCL APIs (the [NVIDIA Collective Communications Library](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)). We provide some starter protocols in `gpu_sim.proto` which simulates single machine CPU and GPU devices with services. This will also help you see the connection between distributed microservices and distributed components on a single machine.

This is an outline for a potential final project and there will not be a fully fleshed-out version of the lab. Some design decisions are intentionally left for you to specify. If you choose this option, you still must submit a project proposal, particularly detailing the problems you intend to solve. This project can be done in a group (following final project group policies).

# Protocol
We model a subset of NCCL, the [NVIDIA Collective Communications Library](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html), which provides APIs for running multi-GPU algorithms.
Please also refer to the [NCCL C API](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in) and the documentation.

In `gpu_sim.proto`, there are three layers, from the bottom to top:
* A single GPU device is simulated by the `GPUDevice` service.
* The NCCL library functions that typically runs on a CPU is simulated by the `GPUCoordinator` service, which calls `GPUDevice` methods to coordinate their communication and computation.
* The application logic for running GPU tasks (e.g., AllReduce, see example below) _uses_ the client of `GPUCoordinator`. (You may build more complicated logic that trains an actual model if you so choose.)

Suggested deployment: each `GPUDevice` server and the `GPUCoordinator` server are run in their respective docker containers.

You may additionally run an existing fault-tolerant key value store (e.g., etcd). It's fine to run it non-replicated and / or in-memory only.

# Bootstrapping
* Each `GPUDevice` should be assigned a unique devide ID (we suggest a random uint64) and listens on a designated port.
  * Each device has its own "memory" address space, and should simulate this memory space with file backing (e.g., mmap-ing or simply writing to a file). E.g., the server can take a "memoryNeeded" parameter and allocates `[minAddr, maxAddr)` internally.
* The `GPUCoordinator` starts with a config file which contains the list of all `GPUDevice`s, their device IDs, IP addresses, and ports.
  * You may choose to build smarter service discovery

# Usage
## Communicator
The Communicator is an NCCL concept, which in NCCL is an object that contains the communication context between multiple GPUs. Within a communicator of N GPUs (N may be less than the total number of GPUs available), each GPU is assigned an integer rank between `[0, N-1]` to identify the role of GPU (e.g., to avoid exposing device IDs).

## Group operations
See [reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html).

## Data transfer
Data loading and transfer is one of the many complexities GPU programming must deal with. See [this blog](https://developer.nvidia.com/blog/machine-learning-frameworks-interoperability-part-2-data-loading-and-data-transfer-bottlenecks/) for details.

In this lab, we simplify the model of data transfer to the following components:
* Between CPU (host) and GPU (device), in either direction;
* Between GPU devices (point-to-point / device-to-device communication).

In practice, to facilitate pipelining and concurrency, NVIDIA GPUs offer the [CUDA](https://developer.nvidia.com/cuda-toolkit) streams abstraction, which is an ordered queue of commands for the GPU devices to asynchronously execute. See [this article](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/) for more details.

In `gpu_sim.proto`, to mimic ncclSend/ncclRecv and cudaStreams, `BeginSend/Receive` initiates device-to-device communication but does not carry data; `StreamSend` is where the data transfer actually happens. This also implies that the GPUDevice service needs to queue up these operations and process the actual data streaming.

You should make explicit decisions on failure assumptions and strategies to handle errors. E.g., what if after a `BeginSend`, GPUDevice A failed to initiate a stream to GPUDevice B? How do you surface that error (and handle it)?

## Other notes
As you look at the NCCL library and our modeling with protobuf, consider the following design questions:
* Why is there a ncclRecv call to begin with?
* What is the difference between a communicator vs. a group?

## Point-to-point communication example
NCCL [reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html).

Send a 64-byte chunk of data from srcMemAddr from one GPU to the dstMemAddr of another GPU.

```
// in GPUCoordinator
{streamId} = comm.gpus[0].BeginSend(srcMemAddr, /* count */ 64, /* dstRank */ 1)
comm.gpus[1].BeginRecv(streamId, dstMemAddr, /* count */ 64, /* srcRank */ 0)

while (comm.gpus[1].GetStreamStatus(streamId) == IN_PROGRESS) {
  // waits for data stream to complete
}

// error handling / GPU 1 now has the data.
```

Actual data is sent from the GPUDevice with rank 0 to GPUDevice with rank 1 using `StreamSend`. You should implement this via grpc client-side streaming (see example [here](https://grpc.io/docs/languages/go/basics/#client-side-streaming-rpc)).

# Part A. Implement the [AllReduceRing](https://github.com/facebookincubator/gloo/blob/main/docs/algorithms.md#allreduce_ring) algorithm.

Goal: so that the following pseudocode block works
```
// N vectors on "CPU / host"
var vecs [][]float64
// fill vectors with data...
client := pb.NewGPUCoordinatorClient(...)

// create / initialize a communicator with N GPUs
{commId, gpus, ...} := client.CommInit(/* numDevices */ N);

// transfer vectors to each individual GPU
for i := 0; i < n; i++ {
  gpu = gpus[i]
  client.Memcpy(vecs[i], gpus[i].deviceId, gpus[i].minAddr)
}

// Initiate the operation
client.GroupStart(commId);
client.AllReduceRing(..., ReduceOp.SUM);
client.GroupEnd(commId);

// check status / synchronize
while (client.GetCommStatus(commId) == IN_PROGRESS) {
  // wait
}
if (client.GetCommStatus(commId) == FAILED) {
  // handle error
}

// data transfer out from GPU 0 to "CPU"
// Since the operation is all-reduce, every GPU has the same data, just pick a GPU as source.
Memcpy(gpus[0].deviceId, gpus.minAddr, numBytes, vecOut)

// Free any resources etc.
CommDestroy();

// output vecOut as final result. vecOut should be equal to the sum of all N vectors.
```

## Evaluation
To show the pros and cons of the Ring algorithm, you may implement a naive version of allReduce for comparison.

Since we want you to be able to develop and run the code on your individual laptops, you may want to simulate large data sizes / inject latency.

# Part B. Crash failure detection and recovery
As is the case with NCCL, if one GPU crashes or slows down (e.g., due to overheating), the operations (the group / communicator) need to be restarted entirely.

Goal: during the above program, if one `GPUDevice` crashes, implement failure detection and ensure the comm does not fail / hang indefinitely.

## Failure detection
For failure detection, we leave the design choices to you. Generally if a `GPUDevice` times out X out the most recent Y requests, one can consider it crashed. You may choose to centralize the healthcheck information, or do decentralized failure detection.

You may also choose to implement online device joining -- after starting a cluster of N GPUs, to be able to register additional devices with the GPUCoordinator which can be used for subsequent operations.

## Failure recovery
You should implement transparent (i.e., automatic) restart with a new group of devices modulo the failed one (e.g., N-1). You may implement checkpointing (e.g., to etcd or a blob store) and restore.

# Other ideas to extend the scope of the project
* multi-communicators (assume disjoint GPU devices)
* Actually train a toy model for MNIST digit classification like [this](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp) or [this](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
