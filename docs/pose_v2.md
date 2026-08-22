# PoseNet V2 — multi-pessoa e associação anatômica

## Objetivos

A V2 foi criada para atacar três limitações observadas na V1:

1. mãos/pés trocando esquerda/direita e formando conexões em X;
2. dificuldade em decidir se realmente existe uma pessoa no frame;
3. incapacidade estrutural de representar duas ou mais pessoas ao mesmo tempo.

## Representação

A V1 recebe uma imagem e retorna 17 heatmaps de uma única pose.

A V2 recebe o frame inteiro e retorna quatro heads em 64x64:

- `center [1,H,W]`: centros de pessoas;
- `keypoints [17,H,W]`: candidatos semânticos de articulações;
- `center_offsets [34,H,W]`: vetor de cada keypoint até o centro da pessoa;
- `parent_offsets [34,H,W]`: vetor de cada keypoint até sua articulação pai.

Um keypoint só entra em um esqueleto quando seu `center_offset` concorda com um
centro detectado. Para cotovelos, punhos, joelhos e tornozelos, o decoder também
usa `parent_offset` para verificar a cadeia anatômica.

Exemplo:

```text
LEFT_WRIST candidato
   -> center_offset deve apontar para Pessoa A
   -> parent_offset deve apontar para LEFT_ELBOW da Pessoa A
```

Se um pico forte de `LEFT_WRIST` estiver na mão direita mas apontar para o
cotovelo direito, ele perde para uma hipótese menor que concorde com o cotovelo
esquerdo. Se não existir alternativa coerente, é preferível ocultar o ponto a
desenhar um X falso.

## Presença e múltiplas pessoas

A presença passa a ser consequência direta do mapa de centros:

```text
0 centros -> 0 pessoas
1 centro  -> 1 pessoa
2 centros -> 2 pessoas
N centros -> N pessoas (até --max-people)
```

O dataset V2 usa imagens COCO completas. Uma imagem sem pessoa utilizável é um
negativo natural: o alvo do mapa de centros e dos keypoints fica zerado.

## Arquitetura e transferência da V1

A V2 preserva os formatos de pesos de:

- encoder1;
- encoder2;
- encoder3;
- bottleneck;
- upsample;
- refine.

A diferença é que a primeira convolução dos encoders usa stride 2, reduzindo a
resolução antes e economizando CPU. O decoder ganha um skip connection em 64x64.

Os heads são novos porque a semântica da saída mudou. O comando `--init-v1`
transfere automaticamente todos os tensores compatíveis.

## Augmentation left/right

O dataset usa flip horizontal semântico. Ao espelhar a imagem ele também troca:

- olhos;
- orelhas;
- ombros;
- cotovelos;
- punhos;
- quadris;
- joelhos;
- tornozelos.

Os vetores X dos offsets também têm o sinal invertido. Isso reduz dependência de
"lado da tela" e reforça lado anatômico.

## Teste antes do treino

```powershell
python -m pytest -v
```

## Smoke test de treino

Use poucos exemplos antes de uma sessão longa:

```powershell
python -m scripts.train_pose_v2 `
  --images data/coco/train2017 `
  --annotations data/coco/annotations/person_keypoints_train2017.json `
  --init-v1 models/pose_model.pt `
  --output models/pose_model_v2_smoke.pt `
  --batch-size 2 `
  --max-samples 500 `
  --max-hours 0.10 `
  --log-every 5
```

Esse checkpoint serve apenas para validar que dataset, transferência e loss
funcionam. Não é esperado que tenha boa qualidade na webcam.

## Primeiro treino real

```powershell
python -m scripts.train_pose_v2 `
  --images data/coco/train2017 `
  --annotations data/coco/annotations/person_keypoints_train2017.json `
  --init-v1 models/pose_model.pt `
  --output models/pose_model_v2.pt `
  --batch-size 4 `
  --epochs 0 `
  --max-hours 6 `
  --learning-rate 0.0001 `
  --log-every 50
```

Para continuar uma V2 já iniciada, substitua `--init-v1` por:

```text
--resume models/pose_model_v2.pt
```

## Webcam V2

Após existir `models/pose_model_v2.pt`:

```powershell
python -m src.main_v2
```

O overlay mostra FPS, quantidade de pessoas, candidatos associados e quantos
pontos bilaterais foram rejeitados para evitar conexões em X.
