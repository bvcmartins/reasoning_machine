#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="reasoning-machine"
IMAGE="localhost/reasoning-machine:latest"
PORT="8503"
ENV_FILE="/home/bmartins/dev/reasoning_machine/.env"

# Stop and remove existing container if present
if podman container exists "$CONTAINER_NAME" 2>/dev/null; then
    echo "Stopping existing $CONTAINER_NAME..."
    podman stop "$CONTAINER_NAME" 2>/dev/null || true
    podman rm "$CONTAINER_NAME" 2>/dev/null || true
fi

echo "Starting $CONTAINER_NAME..."
podman run -d \
    --name "$CONTAINER_NAME" \
    -p "$PORT:$PORT" \
    --add-host=host.docker.internal:host-gateway \
    --env-file "$ENV_FILE" \
    --secret rm_admin_password_hash,type=env,target=ADMIN_PASSWORD_HASH \
    --secret rm_cookie_key,type=env,target=COOKIE_KEY \
    --secret rm_tavily_api_key,type=env,target=TAVILY_API_KEY \
    --restart unless-stopped \
    "$IMAGE"

echo "Waiting for container to be healthy..."
sleep 3

if podman ps --filter "name=$CONTAINER_NAME" --format "{{.Status}}" | grep -q "Up"; then
    echo "$CONTAINER_NAME is running on port $PORT"
else
    echo "ERROR: $CONTAINER_NAME failed to start"
    podman logs "$CONTAINER_NAME"
    exit 1
fi
