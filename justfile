dc := "docker compose"

default: sh

[no-exit-message, no-cd]
sh:
    {{ dc }} exec pi-server bash || true

[no-cd]
build:
    {{ dc }} build pi-server

[no-cd]
restart:
    {{ dc }} restart pi-server

[no-exit-message, no-cd]
logs:
    {{ dc }} logs -f --tail=100 pi-server

[no-cd]
up:
    {{ dc }} up -d pi-server

[no-cd]
down:
    {{ dc }} down pi-server
