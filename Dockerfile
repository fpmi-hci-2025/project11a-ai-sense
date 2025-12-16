FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake wget git ca-certificates libgomp1 \
    libprotobuf-dev protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz && \
    tar -xzf onnxruntime-linux-x64-1.18.0.tgz && \
    rm onnxruntime-linux-x64-1.18.0.tgz

RUN git clone --depth 1 https://github.com/google/sentencepiece.git && \
    cd sentencepiece && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig

WORKDIR /app

RUN mkdir -p libs && \
    wget -q -O libs/httplib.h https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h && \
    wget -q -O libs/json.hpp https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp

COPY . /app

RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.18.0 && \
    cmake --build . --config Release -j$(nproc)

FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/server /app/server
COPY --from=builder /opt/onnxruntime-linux-x64-1.18.0/lib/libonnxruntime.so.1.18.0 /usr/lib/
COPY --from=builder /usr/local/lib/libsentencepiece.so* /usr/lib/

COPY storage/nexus.onnx /app/storage/nexus.onnx
COPY storage/nexus.onnx.data /app/storage/nexus.onnx.data
COPY storage/tokenizer.model /app/storage/tokenizer.model

RUN mkdir -p /app/storage

EXPOSE 7070
CMD ["/app/server"]
