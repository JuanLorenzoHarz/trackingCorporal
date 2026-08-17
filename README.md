# trackingCorporal

Projeto experimental de tracking corporal 2D em tempo real a partir de vídeo de webcam.

## Status

**WIP — Fase 1.**

A estrutura inicial já está preparada e a definição básica do esqueleto corporal já foi implementada. O projeto ainda não faz pose estimation real: por enquanto já existem os 17 keypoints, conexões do esqueleto, tipos compartilhados, renderer básico e um demo artificial que funciona sem webcam.

## Escopo inicial

- uma webcam RGB;
- uma pessoa por vez;
- corpo inteiro em 2D;
- 17 pontos principais do corpo;
- representação por esqueleto virtual;
- tracking temporal e suavização;
- classificação simples das mãos em `OPEN`, `CLOSED` ou `UNKNOWN`.

## Tecnologias planejadas

- Python;
- PyTorch;
- OpenCV;
- NumPy;
- YAML para configurações;
- pytest para testes.

## Direção do projeto

A intenção é estudar e construir o pipeline principal, evitando depender de soluções completas de pose estimation como MediaPipe, OpenPose, MMPose, MoveNet ou YOLO-Pose.

Bibliotecas de infraestrutura, álgebra, visão computacional e deep learning continuam sendo utilizadas normalmente.

## Pipeline planejado

```text
webcam
  -> captura do frame
  -> pré-processamento
  -> estimativa de pose 2D
  -> decodificação dos keypoints
  -> tracking temporal
  -> suavização
  -> regiões das mãos
  -> OPEN / CLOSED / UNKNOWN
  -> visualização
```

## O que já funciona sem webcam

O projeto já consegue representar uma pose artificial com 17 keypoints e desenhar suas conexões em uma imagem usando OpenCV.

Para gerar uma imagem de demonstração:

```powershell
python -m scripts.demo_skeleton
```

O resultado é salvo em:

```text
data/processed/skeleton_demo.png
```

Para gerar e também abrir a imagem em uma janela:

```powershell
python -m scripts.demo_skeleton --show
```

Para executar os testes atuais:

```powershell
python -m pytest -v
```

## Estrutura

```text
configs/       configurações do projeto
src/           código principal
scripts/       treinamento, avaliação e demos
tests/         testes automatizados
data/          dados locais e amostras
models/        pesos/modelos gerados
docs/          decisões e planejamento técnico
```
