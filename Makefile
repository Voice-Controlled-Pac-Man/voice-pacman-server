deploy:
	docker build --platform linux/amd64,linux/arm64 -t europe-central2-docker.pkg.dev/gen-lang-client-0475086233/voice-pacman/server:v1 --push .

run:
	docker run --rm -p 8000:8000 europe-central2-docker.pkg.dev/gen-lang-client-0475086233/voice-pacman/server:v1