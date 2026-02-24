# 使用官方轻量 Python 镜像
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    google-genai \
    google-cloud-firestore \
    "anthropic[vertex]"

# 复制代码
COPY main.py /app/main.py
COPY config.py /app/config.py
COPY models.py /app/models.py
COPY thought_signature_cache.py /app/thought_signature_cache.py
COPY handlers/ /app/handlers/
COPY converters/ /app/converters/

# 环境变量配置
# Cloud Run 会自动设置 GOOGLE_CLOUD_PROJECT
# GOOGLE_CLOUD_LOCATION 默认使用 global
ENV GOOGLE_CLOUD_LOCATION=global
ENV GOOGLE_GENAI_USE_VERTEXAI=true

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["python", "main.py"]