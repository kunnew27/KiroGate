# KiroGate - Docker Image
FROM python:3.11-slim

# 工作目录
WORKDIR /app

# Python 环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（避免部分 pip 包报错）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY kiro_gateway/ ./kiro_gateway/
COPY main.py .

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# 暴露端口（默认 8000，Railway 会通过 PORT 环境变量覆盖）
EXPOSE 8000

# ⚠️【重要】调试阶段先不加 HEALTHCHECK
# 等服务稳定后再加回 /health

# 启动 FastAPI - 使用 shell 形式以支持环境变量替换
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
