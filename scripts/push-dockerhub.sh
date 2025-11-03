#!/bin/bash

# Push script for Haerae Evaluation Toolkit to DockerHub

set -e

# Configuration
DOCKERHUB_USERNAME=${1:-""}
IMAGE_NAME="haerae-evaluation-toolkit"
VERSION=${2:-"latest"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ -z "$DOCKERHUB_USERNAME" ]; then
    echo -e "${RED}Error: DockerHub username is required${NC}"
    echo "Usage: $0 <dockerhub-username> [version]"
    echo "Example: $0 myusername v0.2.0"
    exit 1
fi

DOCKERHUB_IMAGE="${DOCKERHUB_USERNAME}/${IMAGE_NAME}"

echo -e "${GREEN}Preparing to push to DockerHub...${NC}"
echo -e "${YELLOW}DockerHub repository: ${DOCKERHUB_IMAGE}${NC}"

# Check if image exists locally
if ! docker images "${IMAGE_NAME}:latest" --format "{{.Repository}}" | grep -q "${IMAGE_NAME}"; then
    echo -e "${RED}Error: Local image ${IMAGE_NAME}:latest not found${NC}"
    echo "Please build the image first using: ./scripts/build-docker.sh"
    exit 1
fi

# Tag the image for DockerHub
echo -e "${YELLOW}Tagging image for DockerHub...${NC}"
docker tag "${IMAGE_NAME}:latest" "${DOCKERHUB_IMAGE}:${VERSION}"
docker tag "${IMAGE_NAME}:latest" "${DOCKERHUB_IMAGE}:latest"

# Login to DockerHub (if not already logged in)
echo -e "${YELLOW}Checking DockerHub login...${NC}"
if ! docker info | grep -q "Username:"; then
    echo -e "${YELLOW}Please login to DockerHub:${NC}"
    docker login
fi

# Push the images
echo -e "${GREEN}Pushing images to DockerHub...${NC}"
docker push "${DOCKERHUB_IMAGE}:${VERSION}"
docker push "${DOCKERHUB_IMAGE}:latest"

echo -e "${GREEN}Successfully pushed to DockerHub!${NC}"
echo -e "${YELLOW}Images available at:${NC}"
echo "  - ${DOCKERHUB_IMAGE}:${VERSION}"
echo "  - ${DOCKERHUB_IMAGE}:latest"
echo ""
echo -e "${GREEN}To pull and run:${NC}"
echo "docker pull ${DOCKERHUB_IMAGE}:latest"
echo "docker run -it --rm ${DOCKERHUB_IMAGE}:latest"