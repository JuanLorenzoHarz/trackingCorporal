# trackingCorporal

Projeto experimental de tracking corporal 2D em tempo real a partir de webcam.

## Pipeline atual

```text
webcam
 -> recorte quadrado sem distorção
 -> PoseNet
 -> decoder conservador por qualidade de pico
 -> gate de presença corporal
 -> identidade esquerda/direita por par
 -> plausibilidade anatômica
 -> tracking temporal / oclusão curta
 -> smoothing
 -> renderer
```

A inferência padrão segue a regra **na dúvida, não inventar**. Picos secundários não são promovidos automaticamente; o modo experimental antigo continua disponível por `--enable-bilateral-multipeak`.

O gate de presença exige alguns frames com tronco e keypoints coerentes antes de liberar o esqueleto. Quando a pessoa desaparece por tempo suficiente, histórico anatômico, bilateral e smoothing são resetados.

## Testar webcam

```powershell
python -m src.main --mode model
```

Para comparar com a promoção experimental de segundo/terceiro pico:

```powershell
python -m src.main --mode model --enable-bilateral-multipeak
```

## Treino com negativos

Os primeiros treinos usavam apenas recortes contendo pessoas. A rede, portanto, nunca aprendia explicitamente que uma imagem podia não conter pose. O dataset agora pode gerar recortes de fundo quase sem interseção com pessoas anotadas e supervisionar os 17 heatmaps para zero.

Exemplo de fine-tuning conservador:

```powershell
python -m scripts.train_pose `
  --images data/coco/train2017 `
  --annotations data/coco/annotations/person_keypoints_train2017.json `
  --batch-size 8 `
  --epochs 0 `
  --max-hours 6 `
  --learning-rate 0.00008 `
  --resume models/pose_model.pt `
  --output models/pose_model.pt `
  --heatmap-positive-weight 8 `
  --leg-keypoint-weight 1.5 `
  --bilateral-loss-weight 0.01 `
  --bilateral-min-target-distance 3 `
  --negative-sample-probability 0.15 `
  --negative-max-person-overlap 0.01 `
  --occlusion-probability 0.20 `
  --occlusion-min-size 0.12 `
  --occlusion-max-size 0.30 `
  --log-every 50
```

## Testes

```powershell
python -m pytest -v
```

A arquitetura continua sendo a PoseNet própria do projeto; nenhuma solução pronta de pose foi introduzida.
