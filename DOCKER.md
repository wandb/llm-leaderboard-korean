# Docker Guide for Haerae Evaluation Toolkit

This guide explains how to build, run, and deploy the Haerae Evaluation Toolkit using Docker.

## Quick Start

### 1. Build the Docker Image

```bash
# Build with default settings
./scripts/build-docker.sh

# Build with specific version
./scripts/build-docker.sh v0.2.0

# Build with registry prefix
./scripts/build-docker.sh latest myregistry.com
```

### 2. Run the Container

```bash
# Basic run
docker run -it --rm haerae-evaluation-toolkit:latest

# Run with environment variables
docker run -it --rm \
  -e OPENAI_API_KEY=your_key_here \
  -e WANDB_API_KEY=your_wandb_key \
  haerae-evaluation-toolkit:latest

# Run with volume mounts
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  -p 8000:8000 \
  haerae-evaluation-toolkit:latest
```

### 3. Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Building and Pushing to DockerHub

### Prerequisites

1. Docker installed and running
2. DockerHub account
3. Logged into DockerHub: `docker login`

### Build and Push

```bash
# 1. Build the image
./scripts/build-docker.sh v0.2.0

# 2. Push to DockerHub
./scripts/push-dockerhub.sh your-dockerhub-username v0.2.0
```

### Manual Push Process

```bash
# Tag for DockerHub
docker tag haerae-evaluation-toolkit:latest your-username/haerae-evaluation-toolkit:latest
docker tag haerae-evaluation-toolkit:latest your-username/haerae-evaluation-toolkit:v0.2.0

# Push to DockerHub
docker push your-username/haerae-evaluation-toolkit:latest
docker push your-username/haerae-evaluation-toolkit:v0.2.0
```

## Environment Variables

The container supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `WANDB_API_KEY` | Weights & Biases API key | - |
| `HRET_HOME` | Application home directory | `/app` |
| `PYTHONPATH` | Python path | `/app` |

## Volume Mounts

Recommended volume mounts:

- `/app/data` - For datasets and evaluation results
- `/app/logs` - For application logs
- `/app/configs` - For configuration files

## Ports

- `8000` - FastAPI web server (if running web interface)

## Docker Compose Configuration

The `docker-compose.yml` includes:

- **hret**: Main application container
- **redis**: Optional Redis cache (commented out by default)

### Environment File

Create a `.env` file in the project root:

```env
WANDB_API_KEY=your_wandb_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Advanced Usage

### Development Mode

For development, you can mount the source code:

```bash
docker run -it --rm \
  -v $(pwd):/app \
  -e PYTHONPATH=/app \
  haerae-evaluation-toolkit:latest \
  bash
```

### Custom Entrypoint

```bash
# Run specific evaluation
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  haerae-evaluation-toolkit:latest \
  python run_eval.py --config configs/your_config.yaml

# Start web server
docker run -it --rm \
  -p 8000:8000 \
  haerae-evaluation-toolkit:latest \
  uvicorn llm_eval.api:app --host 0.0.0.0 --port 8000
```

### Multi-platform Build

For building multi-platform images:

```bash
# Enable buildx
docker buildx create --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag your-username/haerae-evaluation-toolkit:latest \
  --push .
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase Docker memory limit
2. **Permission Issues**: Ensure proper file permissions
3. **API Keys**: Verify environment variables are set correctly

### Debug Mode

```bash
# Run with debug output
docker run -it --rm \
  -e PYTHONUNBUFFERED=1 \
  -e LOG_LEVEL=DEBUG \
  haerae-evaluation-toolkit:latest
```

### Container Shell Access

```bash
# Access running container
docker exec -it hret-container bash

# Run new container with shell
docker run -it --rm haerae-evaluation-toolkit:latest bash
```

## Image Details

- **Base Image**: `python:3.11-slim`
- **Architecture**: Multi-stage build for optimization
- **Size**: Approximately 2-3GB (due to ML dependencies)
- **User**: Non-root user `hret` for security

## Security Considerations

- Container runs as non-root user
- Sensitive data should be passed via environment variables
- Use secrets management for production deployments
- Regular image updates for security patches

## Performance Tips

1. Use volume mounts for large datasets
2. Configure appropriate memory limits
3. Use Redis for caching (included in docker-compose)
4. Consider GPU support for model inference

## Support

For issues related to Docker deployment, please check:

1. Docker logs: `docker logs container-name`
2. Application logs in `/app/logs`
3. GitHub issues for known problems