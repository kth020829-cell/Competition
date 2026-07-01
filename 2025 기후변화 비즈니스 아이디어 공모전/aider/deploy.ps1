# ============================================================
# Aider AWS 배포 스크립트 (PowerShell)
# 실행: ./deploy.ps1
# 전제: AWS CLI, SAM CLI, Docker Desktop 설치 완료
# ============================================================

param(
    [string]$Region     = "ap-northeast-2",
    [string]$StackName  = "aider-stack",
    [string]$KMAKey     = "",
    [string]$AirKey     = ""
)

$ErrorActionPreference = "Stop"

# 네이티브 명령(docker, sam, npm 등)은 $ErrorActionPreference 가 Stop 이어도
# 자동으로 예외를 던지지 않으므로, 각 명령 직후 이 함수로 exit code 를 명시 검사한다.
function Assert-ExitCode {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] $Step (exit code $LASTEXITCODE)"
    }
}

# ── 0. 계정 ID 확인 ─────────────────────────────────────────
Write-Host "`n[1/6] AWS 계정 확인..."
$AccountId = (aws sts get-caller-identity --query Account --output text)
Assert-ExitCode "AWS 자격증명 확인"
Write-Host "  AccountId: $AccountId, Region: $Region"

$EcrRepo = "$AccountId.dkr.ecr.$Region.amazonaws.com/aider-backend"

# ── 1. ECR 레포 확인/생성 ──────────────────────────────────
# 이 블록만 "Continue" 로 낮춰 non-zero exit 를 $LASTEXITCODE 로 처리한다.
Write-Host "`n[2/6] ECR 레포 확인/생성..."
$ErrorActionPreference = "Continue"
aws ecr describe-repositories --repository-names aider-backend --region $Region | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  레포가 없습니다. 새로 생성합니다..."
    aws ecr create-repository --repository-name aider-backend --region $Region | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $ErrorActionPreference = "Stop"
        throw "[FAIL] ECR 레포 생성 실패. IAM 권한(ecr:CreateRepository)을 확인하세요."
    }
    Write-Host "  ECR 레포 생성 완료."
} else {
    Write-Host "  ECR 레포가 이미 존재합니다. 재사용합니다."
}
$ErrorActionPreference = "Stop"
Write-Host "  ECR: $EcrRepo"

# ── 2. Docker 이미지 빌드 & 푸시 ───────────────────────────
Write-Host "`n[3/6] Docker 이미지 빌드 중 (torch 포함으로 5~10분 소요)..."
Set-Location "$PSScriptRoot\backend"

if (-not (Test-Path "artifacts\lgbm_chai.pkl")) {
    Write-Host "  [!] artifacts/ 가 비어있습니다. 먼저 학습 스크립트를 실행하세요:"
    Write-Host "      .venv\Scripts\python training\train_lgbm.py"
    Write-Host "      .venv\Scripts\python training\train_autoencoder.py"
    Write-Host "      .venv\Scripts\python training\train_transformer.py"
    exit 1
}

docker build --provenance=false --sbom=false -f Dockerfile.lambda -t aider-backend:latest .
Assert-ExitCode "Docker 이미지 빌드"

Write-Host "`n  ECR 로그인..."
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin "$AccountId.dkr.ecr.$Region.amazonaws.com"
Assert-ExitCode "ECR 로그인"

$ImageTag = (Get-Date -Format "yyyyMMdd-HHmmss")
docker tag aider-backend:latest "${EcrRepo}:${ImageTag}"
docker tag aider-backend:latest "${EcrRepo}:latest"
docker push "${EcrRepo}:${ImageTag}"
Assert-ExitCode "ECR 이미지 푸시"
docker push "${EcrRepo}:latest"
Assert-ExitCode "ECR latest 태그 푸시"
Write-Host "  이미지 푸시 완료: ${EcrRepo}:${ImageTag}"

Set-Location "$PSScriptRoot"

# ── 3. SAM 배포 ────────────────────────────────────────────
Write-Host "`n[4/6] SAM 배포 (CloudFormation 스택 생성/업데이트)..."
sam deploy `
    --template-file infra/template.yaml `
    --stack-name $StackName `
    --region $Region `
    --capabilities CAPABILITY_IAM `
    --image-repository "$AccountId.dkr.ecr.$Region.amazonaws.com/aider-backend" `
    --parameter-overrides `
        "EcrImageUri=${EcrRepo}:${ImageTag}" `
        "KMAApiKey=$KMAKey" `
        "AirKoreaApiKey=$AirKey" `
    --no-fail-on-empty-changeset
Assert-ExitCode "SAM 배포"

# ── 4. API 엔드포인트 가져오기 ──────────────────────────────
Write-Host "`n[5/6] 배포 결과 확인..."
$ApiUrl = (aws cloudformation describe-stacks `
    --stack-name $StackName `
    --region $Region `
    --query "Stacks[0].Outputs[?OutputKey=='ApiEndpoint'].OutputValue" `
    --output text)
Assert-ExitCode "CloudFormation 스택 조회 (ApiEndpoint)"

$BucketName = (aws cloudformation describe-stacks `
    --stack-name $StackName `
    --region $Region `
    --query "Stacks[0].Outputs[?OutputKey=='FrontendBucketName'].OutputValue" `
    --output text)
Assert-ExitCode "CloudFormation 스택 조회 (FrontendBucketName)"

$CfUrl = (aws cloudformation describe-stacks `
    --stack-name $StackName `
    --region $Region `
    --query "Stacks[0].Outputs[?OutputKey=='CloudFrontUrl'].OutputValue" `
    --output text)
Assert-ExitCode "CloudFormation 스택 조회 (CloudFrontUrl)"

$DistributionId = (aws cloudformation describe-stacks `
    --stack-name $StackName `
    --region $Region `
    --query "Stacks[0].Outputs[?OutputKey=='FrontendDistributionId'].OutputValue" `
    --output text)
Assert-ExitCode "CloudFormation 스택 조회 (FrontendDistributionId)"

Write-Host "  API 엔드포인트: $ApiUrl"
Write-Host "  S3 버킷: $BucketName"
Write-Host "  CloudFront URL: $CfUrl"
Write-Host "  CloudFront Distribution ID: $DistributionId"

# ── 5. 프론트엔드 빌드 & S3 업로드 ────────────────────────
Write-Host "`n[6/6] 프론트엔드 빌드 및 S3 업로드..."

# .nvmrc 에 지정된 Node 18 으로 전환
# nvm-windows 는 심볼릭 링크(NVM_SYMLINK)를 교체하므로 이후 PATH 에 반영된다.
$NvmExe = "$env:APPDATA\..\Local\nvm\nvm.exe"
if (-not (Test-Path $NvmExe)) { $NvmExe = "C:\Users\$env:USERNAME\AppData\Local\nvm\nvm.exe" }
$env:NVM_HOME    = Split-Path $NvmExe
$env:NVM_SYMLINK = "C:\nvm4w\nodejs"

Write-Host "  Node 버전 전환: nvm use 18.20.4"
$ErrorActionPreference = "Continue"
& $NvmExe use 18.20.4
if ($LASTEXITCODE -ne 0) {
    $ErrorActionPreference = "Stop"
    throw "[FAIL] nvm use 18.20.4 실패. 먼저 `nvm install 18.20.4` 를 실행하세요."
}
$ErrorActionPreference = "Stop"

# nvm use 가 심볼릭 링크를 교체했으므로 PATH 에 반영
$env:PATH = "$env:NVM_SYMLINK;$env:PATH"
Write-Host "  Node: $(node --version), npm: $(npm --version)"

Set-Location "$PSScriptRoot\frontend"

# 배포마다 바뀌는 API Gateway 주소를 .env.production 에 자동 반영 (하드코딩 금지)
# CRA 는 프로덕션 빌드 시 이 파일을 자동으로 읽어 REACT_APP_API_URL 을 번들에 굽는다.
Set-Content -Path ".env.production" -Value "REACT_APP_API_URL=$ApiUrl" -Encoding utf8NoBOM
Write-Host "  .env.production 갱신: REACT_APP_API_URL=$ApiUrl"

# ajv-keywords 버전 충돌로 fork-ts-checker-webpack-plugin 이 crash 하는 경우 우회
$env:DISABLE_ESLINT_PLUGIN  = "true"
$env:TSC_COMPILE_ON_ERROR   = "true"

npm install --legacy-peer-deps --silent
Assert-ExitCode "npm install"

npm run build
Assert-ExitCode "npm run build"

aws s3 sync build/ "s3://$BucketName/" --delete
Assert-ExitCode "S3 업로드"

Write-Host "  CloudFront 캐시 무효화 중..."
aws cloudfront create-invalidation --distribution-id $DistributionId --paths "/*" | Out-Null
Assert-ExitCode "CloudFront 캐시 무효화"

Set-Location "$PSScriptRoot"

Write-Host "`n============================================================"
Write-Host "배포 완료!"
Write-Host "  백엔드 API : $ApiUrl"
Write-Host "  프론트엔드 : $CfUrl"
Write-Host "  API 문서   : $ApiUrl/docs"
Write-Host "============================================================`n"
