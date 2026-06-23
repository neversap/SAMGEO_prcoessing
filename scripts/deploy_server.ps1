param(
    [Parameter(Mandatory = $true)]
    [string]$Host,

    [string]$RemotePath = "/opt/sam-geo"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath "sam3-main.zip")) {
    throw "sam3-main.zip was not found in the project root."
}

ssh $Host "mkdir -p '$RemotePath'"
scp -r `
    .env.example `
    Dockerfile.server `
    Dockerfile.server.baked-model `
    docker-compose.baked-model.yml `
    docker-compose.yml `
    requirements-sam3.txt `
    requirements-server.txt `
    sam3-main.zip `
    docs `
    scripts `
    server `
    "$Host`:$RemotePath/"

Write-Host "Uploaded SAM GEO server files to $Host`:$RemotePath"
Write-Host "Next: ssh $Host 'cd $RemotePath && cp -n .env.example .env && docker compose up -d --build'"
