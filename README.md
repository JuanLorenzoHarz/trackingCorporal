# trackingCorporal

Projeto experimental de tracking corporal 2D em tempo real a partir de vídeo de webcam.

## Status

**WIP — Fase 1: estruturação e preparação do projeto.**

Ainda não há implementação do sistema de tracking. Nesta etapa o objetivo é separar responsabilidades, preparar o ambiente e definir o pipeline antes de começar a desenvolver os algoritmos.

## Escopo inicial

- uma webcam RGB;
- uma pessoa por vez;
- corpo inteiro em 2D;
- aproximadamente 17 pontos principais do corpo;
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

## Estrutura

```text
configs/       configurações do projeto
src/           código principal
scripts/       treinamento e avaliação
tests/         testes automatizados
data/          dados locais e amostras
models/        pesos/modelos gerados
docs/          decisões e planejamento técnico
```
