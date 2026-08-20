# trackingCorporal

Projeto experimental de tracking corporal 2D em tempo real a partir de vídeo de webcam.

## Status

**WIP — Fase 1.**

O projeto já possui os 17 keypoints, renderer, captura de webcam, loop ao vivo, primeira CNN própria de pose por heatmaps, tracking temporal, previsão curta de keypoints ocultos e suavização.

O modo `demo` continua disponível com uma pose fixa. O modo `model` usa a CNN treinada e passa a pose por um tracker temporal antes de desenhar o esqueleto.

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
  -> 17 keypoints observados
  -> tracker temporal
       -> velocidade recente
       -> geometria dos membros
       -> previsão curta de pontos ocultos
       -> confidence decay
  -> suavização exponencial
  -> renderer
```

Quando a CNN perde temporariamente uma articulação, o tracker tenta mantê-la usando o histórico e a relação geométrica previamente observada com articulações vizinhas. A confiança cai a cada frame previsto e o ponto expira após um limite, evitando manter uma pose inventada indefinidamente.

## Testes

```powershell
python -m pytest -v
```

Existem testes para câmera, dataset COCO, heatmaps, CNN, decoder, renderer, previsão temporal, expiração de pontos ocultos, suavização e oclusão artificial.

## Webcam — modo demo

```powershell
python -m src.main
```

ou explicitamente:

```powershell
python -m src.main --mode demo
```

Esse modo mostra a webcam com uma pose artificial fixa e a mensagem `POSE DEMO - NOT TRACKING`.

## Webcam — modelo + tracking temporal

Depois que `models/pose_model.pt` existir:

```powershell
python -m src.main --mode model
```

O padrão atual usa:

```text
confidence = 0.15
prediction frames = 8
prediction decay = 0.82
anatomy weight = 0.60
smoothing alpha = 0.65
```

Esses valores podem ser ajustados pela linha de comando:

```powershell
python -m src.main --mode model `
  --confidence 0.15 `
  --prediction-frames 8 `
  --prediction-decay 0.82 `
  --anatomy-weight 0.60 `
  --smoothing-alpha 0.65
```

A janela exibe também `Keypoints previstos`, indicando quantas articulações não estão sendo observadas diretamente pela CNN naquele frame e estão sendo mantidas pelo tracker.

## Preparar dados COCO

O projeto usa as anotações de keypoints de pessoas do COCO 2017. Os dados ficam em `data/coco/` e são ignorados pelo Git.

Validação:

```powershell
python -m scripts.download_coco --split val
```

Treino:

```powershell
python -m scripts.download_coco --split train
```

## Treino controlado por tempo

Exemplo de treino por 8 horas:

```powershell
python -m scripts.train_pose `
  --images data/coco/train2017 `
  --annotations data/coco/annotations/person_keypoints_train2017.json `
  --batch-size 8 `
  --epochs 0 `
  --max-hours 8 `
  --output models/pose_model.pt
```

Para continuar um modelo existente:

```powershell
python -m scripts.train_pose `
  --images data/coco/train2017 `
  --annotations data/coco/annotations/person_keypoints_train2017.json `
  --batch-size 8 `
  --epochs 0 `
  --max-hours 4 `
  --learning-rate 0.0005 `
  --resume models/pose_model.pt `
  --output models/pose_model.pt
```

## Fine-tuning para oclusões

O dataset pode esconder artificialmente uma região da pessoa **sem remover os targets dos keypoints**. Assim a CNN é treinada a inferir articulações a partir do restante do corpo.

Exemplo:

```powershell
python -m scripts.train_pose `
  --images data/coco/train2017 `
  --annotations data/coco/annotations/person_keypoints_train2017.json `
  --batch-size 8 `
  --epochs 0 `
  --max-hours 3 `
  --learning-rate 0.0003 `
  --resume models/pose_model.pt `
  --output models/pose_model.pt `
  --occlusion-probability 0.35 `
  --occlusion-min-size 0.12 `
  --occlusion-max-size 0.35 `
  --log-every 50
```

`--occlusion-probability 0.35` significa que aproximadamente 35% dos exemplos recebidos pela rede terão uma região escondida artificialmente. Os heatmaps corretos continuam presentes como alvo.

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
