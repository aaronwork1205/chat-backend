FROM golang:1.21 as builder

# 安装依赖
RUN apt-get update && apt-get install -y wget unzip build-essential

# 安装 ONNX Runtime C API
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz && \
    tar -xzf onnxruntime-linux-x64-1.17.0.tgz && \
    mv onnxruntime-linux-x64-1.17.0 /onnx

ENV CGO_CFLAGS="-I/onnx/include"
ENV CGO_LDFLAGS="-L/onnx/lib -lonnxruntime"

WORKDIR /app
COPY . .
RUN go build -o server ./cmd/server

# runtime image
FROM debian:bullseye-slim

COPY --from=builder /app/server /app/server
COPY --from=builder /onnx /onnx
COPY model/ /app/model

ENV LD_LIBRARY_PATH="/onnx/lib"
WORKDIR /app
CMD ["./server"]
