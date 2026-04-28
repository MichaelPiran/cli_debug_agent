BINARY_NAME := cli-debug-agent
BINARY_PATH := bin/$(BINARY_NAME)

.PHONY: build clean run

build:
	@mkdir -p bin
	go build -o $(BINARY_PATH) .

run: build
	./$(BINARY_PATH) --help

clean:
	rm -rf bin