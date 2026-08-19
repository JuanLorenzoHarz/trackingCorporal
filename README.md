# trackingCorporal

Projeto experimental de tracking corporal 2D em tempo real a partir de vídeo de webcam.

## Status

**WIP — Fase 1.**

A estrutura inicial já está preparada. O projeto possui os 17 keypoints, conexões do esqueleto, tipos compartilhados, renderer, captura de webcam, loop de vídeo ao vivo e a primeira CNN própria de estimativa de pose por heatmaps.

O modo `demo` continua disponível com uma pose fixa. O modo `model` usa a CNN treinada e já produz keypoints quadro a quadro, mas ainda não possui tracking temporal nem suavização.

## Escopo inicial

- uma webcam RGB;
- uma pessoa por vez;
- corpo inteiro em 2D;
- 17 pontos principais do corpo;
- representação por esqueleto virtual;
- tracking temporal e suavização;
- classificação simples das mãos em `OPEN`, `CLOSED` ou `UNKNOWN`.

## Tecnologias

- Python;
- PyTorch;
- OpenCV;
- NumPy;
- YAML para configurações;
- pytest para testes.

## Direção do projeto

A intenção é estudar e construir o pipeline principal, evitando depender de soluções completas de pose estimation como MediaPipe, OpenPose, MMPose, MoveNet ou YOLO-Pose.

A primeira rede (`PoseNet`) é uma CNN encoder-decoder criada no próprio projeto. Ela recebe uma imagem RGB 256x256 e produz 17 heatmaps 64x64.

## Pipeline atual

```text
webcam
  -> captura do frame
  -> resize/normalização
  -> PoseNet
  -> 17 heatmaps
  -> decoder
  -> 17 keypoints
  -> renderer
```

Ainda faltam tracking temporal, suavização e classificação das mãos.

## Testes

```powershell
python -m pytest -v
```

Os testes de `Camera` usam mocks e não exigem webcam física. Também existem testes para shape da CNN, heatmaps e decoder.

## Webcam — modo demo

```powershell
python -m src.main
```

ou explicitamente:

```powershell
python -m src.main --mode demo
```

Esse modo mostra a webcam com uma pose artificial fixa e a mensagem `POSE DEMO - NOT TRACKING`.

## Preparar dados COCO

O projeto usa as anotações de keypoints de pessoas do COCO 2017. Os dados ficam em `data/coco/` e são ignorados pelo Git.

Para baixar primeiro o conjunto de validação, menor e útil para validar o pipeline:

```powershell
python -m scripts.download_coco --split val
```

Isso prepara:

```text
data/coco/
├── val2017/
└── annotations/
    └── person_keypoints_val2017.json
```

Para o treino real, baixe o conjunto de treinamento:

```powershell
python -m scripts.download_coco --split train
```

O conjunto `train2017` é grande; não é necessário baixá-lo apenas para verificar se o código funciona.

## Smoke test de treinamento

Depois de baixar `val`, dá para validar todo o caminho de treino com poucos exemplos:

```powershell
python -m scripts.train_pose `
  --images data/coco/val2017 `
  --annotations data/coco/annotations/person_keypoints_val2017.json `
  --max-samples 128 `
  --batch-size 8 `
  --epochs 1
```

Esse comando serve apenas para provar que dataset -> heatmaps -> CNN -> loss -> checkpoint funciona. Ele não produz um modelo de boa qualidade.

## Treino real

Depois de baixar `train2017`:

```powershell
python -m scripts.train_pose `
  --images data/coco/train2017 `
  --annotations data/coco/annotations/person_keypoints_train2017.json `
  --batch-size 16 `
  --epochs 10
```

O checkpoint é salvo em:

```text
models/pose_model.pt
```

Treinar por CPU é possível, mas pode ser lento. A quantidade de épocas, batch size e arquitetura serão ajustadas com base nos primeiros resultados.

## Webcam — pose estimada pela CNN

Depois que `models/pose_model.pt` existir:

```powershell
python -m src.main --mode model
```

Para alterar o limiar de confiança:

```powershell
python -m src.main --mode model --confidence 0.25
```

Nesta primeira versão, a inferência assume **uma pessoa centralizada e ocupando boa parte do frame**. A CNN estima pose quadro a quadro; isso ainda não é tracking temporal.

## Demo sem webcam

```powershell
python -m scripts.demo_skeleton --show
```

O resultado também é salvo em:

```text
data/processed/skeleton_demo.png
```

## Estrutura

```text
configs/       configurações do projeto
src/           código principal
scripts/       treinamento, download, avaliação e demos
tests/         testes automatizados
data/          datasets e dados locais (não versionados)
models/        pesos/modelos gerados (não versionados)
docs/          decisões e planejamento técnico
```
