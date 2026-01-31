#!/bin/bash

if [ -z "$GH_TOKEN" ]; then
    echo "You need to specify a GitHub token in GH_TOKEN env variable!"
    exit 1
fi;

set -ex

# build everything into one docker container
docker build --rm --build-arg GH_TOKEN=${GH_TOKEN} -t pbl .

# publish docker container on GH
if [ "$TRAVIS_BRANCH" = "master" -a "$TRAVIS_PULL_REQUEST" = "false" ]; then# 
    export IMAGE_ID=`docker images -q pbl`
    echo "$GH_TOKEN" | docker login docker.pkg.github.com --username b52 --password-stdin
    docker tag $IMAGE_ID docker.pkg.github.com/fhkiel-mlaip/pbl/pbl:latest
    docker push --max-concurrent-uploads=1 docker.pkg.github.com/fhkiel-mlaip/pbl/pbl:latest
fi;
