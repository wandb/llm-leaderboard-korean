#!/bin/bash

# Build script for Haerae Evaluation Toolkit Docker image

set -e

# Configuration
IMAGE_NAME="haerae-evaluation-toolkit"
VERSION=${1:-"latest"}
REGISTRY=${2:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Haerae Evaluation Toolkit Docker image...${NC}"

# Build the image
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}:${VERSION}"
fi

echo -e "${YELLOW}Building image: ${FULL_IMAGE_NAME}${NC}"

docker build \
    --tag "${FULL_IMAGE_NAME}" \
    --tag "${IMAGE_NAME}:latest" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
    --build-arg VERSION="${VERSION}" \
    .

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${YELLOW}Image: ${FULL_IMAGE_NAME}${NC}"

# Show image size
echo -e "${YELLOW}Image size:${NC}"
docker images "${FULL_IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo -e "${GREEN}To run the container:${NC}"
echo "docker run -it --rm ${FULL_IMAGE_NAME}"
echo ""
echo -e "${GREEN}To push to registry (if configured):${NC}"
echo "docker push ${FULL_IMAGE_NAME}"