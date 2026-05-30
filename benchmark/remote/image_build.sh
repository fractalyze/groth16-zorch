#!/usr/bin/env bash
#
# Build a rabbit benchmark image for a given GPU target and (optionally)
# push it to ghcr.io. Pairs with the Dockerfile.<target> + the matching
# bench_config.<target-tag>.yaml in this directory.
#
# Usage:
#   benchmark/remote/image_build.sh <target> build         # build, no push
#   benchmark/remote/image_build.sh <target> build --push  # build + push
#   benchmark/remote/image_build.sh <target> push          # push only
#
#   <target> is one of: rocm | cuda
#
# The image tag tracks the rabbit HEAD sha so the plugin can pin a
# specific build for a measurement run.

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <rocm|cuda> {build [--push] | push}" >&2
  exit 2
fi

TARGET="$1"; shift
case "$TARGET" in
  rocm|cuda) ;;
  *) echo "Unknown target: $TARGET (expected rocm | cuda)" >&2; exit 2 ;;
esac

REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$REPO_ROOT"

REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE="${IMAGE:-fractalyze/rabbitsnark-py}"
SHA=$(git rev-parse --short HEAD)
TAG_SHA="rabbit-${TARGET}-${SHA}"
TAG_LATEST="rabbit-${TARGET}-latest"

FULL_SHA="${REGISTRY}/${IMAGE}:${TAG_SHA}"
FULL_LATEST="${REGISTRY}/${IMAGE}:${TAG_LATEST}"
DOCKERFILE="benchmark/remote/Dockerfile.${TARGET}"

cmd="${1:-build}"
shift || true

build() {
  local push=0
  for arg in "$@"; do
    case "$arg" in
      --push) push=1 ;;
      *) echo "Unknown build arg: $arg" >&2; exit 2 ;;
    esac
  done

  DOCKER_BUILDKIT=1 docker build \
    -f "$DOCKERFILE" \
    -t "$FULL_SHA" \
    -t "$FULL_LATEST" \
    .

  if [ "$push" = "1" ]; then
    push_impl
  else
    echo
    echo "Built:"
    echo "  $FULL_SHA"
    echo "  $FULL_LATEST"
    echo "Push with: $0 $TARGET push"
  fi
}

push_impl() {
  # `docker login ghcr.io` must have happened separately (CI uses
  # GITHUB_TOKEN; locally the user runs login once).
  docker push "$FULL_SHA"
  docker push "$FULL_LATEST"
  echo
  echo "Pushed:"
  echo "  $FULL_SHA"
  echo "  $FULL_LATEST"
}

case "$cmd" in
  build) build "$@" ;;
  push)  push_impl ;;
  *)
    echo "Usage: $0 $TARGET {build [--push] | push}" >&2
    exit 2
    ;;
esac
