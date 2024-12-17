# dsml

# Running the Project 

run steps: go mod tidy
go run cmd/device/main.go --port 50051
go run cmd/device/main.go --port 50052

to test
go test ./pkg/tests -v


# Group Work
As a suite, we all worked concurrently on this project on one laptop. We worked together to implement the RPC calls and building out the services. Then, we moved on to working on the AllReduceRing and NaiveAllReduce implementations. Throughout the lab, we worked together to implement unit testing to ensure that our implementation was working as intended. Finally, we worked on implementing failure recovery and completed the write-up.

