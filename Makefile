.PHONY: build run scan push clean

IMAGE_NAME = anti-gravity-api
TAG = latest
PORT = 8080

build:
	docker build -t $(IMAGE_NAME):$(TAG) .

run:
	docker run -p $(PORT):8080 \
		-e MODEL_URI="models:/anti_gravity_v1/Production" \
		-e MLFLOW_TRACKING_URI="http://host.docker.internal:5000" \
		-e PORT=8080 \
		-e WORKERS=4 \
		--name $(IMAGE_NAME)-container \
		--rm $(IMAGE_NAME):$(TAG)

scan:
	trivy image $(IMAGE_NAME):$(TAG)

push:
	docker tag $(IMAGE_NAME):$(TAG) your-registry.com/$(IMAGE_NAME):$(TAG)
	docker push your-registry.com/$(IMAGE_NAME):$(TAG)

clean:
	docker rmi $(IMAGE_NAME):$(TAG) --force
