FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# 기본 패키지 설치 및 타임존 설정 우회
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 3.11 설치
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# pip 및 uv 설치
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# 작업 디렉토리 설정 및 레포지토리 복사
WORKDIR /workspace
RUN git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git

# 파이토치 및 알파베타크라운 의존성 설치
WORKDIR /workspace/alpha-beta-CROWN
RUN uv pip install --system --reinstall torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu126
RUN uv pip install --system .

# 테스트 스크립트 및 설정 파일 위치 지정
WORKDIR /workspace
RUN mkdir -p models configs data
COPY test.py /workspace/test.py
COPY configs/mnist_verification.yaml /workspace/configs/mnist_verification.yaml

# 컨테이너 실행 시 기본 명령어
CMD ["python", "test.py"]
