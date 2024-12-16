# dsml

run steps: go mod tidy
go run cmd/device/main.go --port 50051
go run cmd/device/main.go --port 50052

to test
go test ./pkg/tests -v