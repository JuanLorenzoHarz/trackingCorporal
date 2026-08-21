# trackingCorporal

Projeto experimental de tracking corporal 2D em tempo real a partir de vídeo de webcam.

## Status

**WIP — Fase 1.**

O pipeline atual possui captura de webcam, CNN própria por heatmaps, pré-processamento sem distorção, resolução bilateral de pernas, filtro de plausibilidade, tracking temporal e suavização.

## Escopo inicial

- uma webcam RGB;
- uma pessoa por vez;
- corpo inteiro em 2D;
- 17 keypoints COCO;
- esqueleto virtual;
- tracking temporal e suavização;
- previsão curta de articulações ocultas;
- classificação futura das mãos em `OPEN`, `CLOSED` ou `UNKNOWN`.

## Pipeline atual

```text
webcam
  -> recorte quadrado sem distorção
  -> PoseNet
  -> 17 heatmaps
  -> decoder com múltiplas hipóteses L/R para joelhos/tornozelos
  -> estabilização temporal da identidade das pernas
  -> plausibilidade anatômica/temporal
  -> tracking de oclusões
  -> smoothing
  -> renderer
```

A resolução bilateral tenta evitar que os canais esquerdo e direito colapsem sobre a mesma perna. Quando os maiores picos de joelho/tornozelo ficam juntos, o decoder verifica a segunda e terceira máximas locais e só escolhe uma alternativa quando sua confiança continua próxima do pico principal. Em seguida, o tracker bilateral usa o histórico para reduzir trocas de identidade entre esquerda e direita.

## Testes

```powershell
python -m pytest -v
```

## Webcam

```powershell
python -m src.main --mode model
```

O overlay mostra FPS, plausibilidade geral, plausibilidade das pernas, consistência L/R, correções de picos, colapsos detectados, trocas de identidade, keypoints rejeitados/previstos e calibração anatômica.

## Fine-tuning focado em pernas

O treino suporta três mecanismos complementares:

- `--heatmap-positive-weight`: dá mais peso ao pico da articulação;
- `--leg-keypoint-weight`: aumenta a importância de quadris, joelhos e tornozelos;
- `--bilateral-loss-weight`: penaliza resposta do canal esquerdo sobre o target direito e vice-versa quando os targets estão claramente separados.

Exemplo:

```powershell
python -m scripts.train_pose `
  --images data/coco/train2017 `
  --annotations data/coco/annotations/person_keypoints_train2017.json `
  --batch-size 8 `
  --epochs 0 `
  --max-hours 6 `
  --learning-rate 0.00015 `
  --resume models/pose_model.pt `
  --output models/pose_model.pt `
  --heatmap-positive-weight 8 `
  --leg-keypoint-weight 2.0 `
  --bilateral-loss-weight 0.02 `
  --bilateral-min-target-distance 3 `
  --occlusion-probability 0.30 `
  --occlusion-min-size 0.12 `
  --occlusion-max-size 0.30 `
  --log-every 50
```

A loss bilateral não força esquerda/direita a permanecerem em lados fixos da tela. Ela só entra quando o ground truth possui os dois pontos suficientemente separados.

## Tecnologias

- Python;
- PyTorch;
- OpenCV;
- NumPy;
- PyYAML;
- pytest.

A intenção do projeto é construir e entender o pipeline principal sem depender de soluções completas como MediaPipe, OpenPose, MMPose, MoveNet ou YOLO-Pose.
