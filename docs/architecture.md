# Arquitetura inicial

## Objetivo

Manter o pipeline dividido em componentes pequenos para permitir estudar, substituir e medir cada etapa independentemente.

## Fluxo planejado

```text
Camera
  -> FramePreprocessor
  -> PoseModel
  -> PoseDecoder
  -> TemporalTracker
  -> Smoothing
  -> HandROI
  -> HandClassifier
  -> Renderer
```

## Separação de responsabilidades

### Captura
Obtém frames da webcam. Não conhece modelos de machine learning.

### Pré-processamento
Prepara a imagem para o modelo e guarda informações necessárias para converter coordenadas de volta ao frame original.

### Pose
Contém a definição do esqueleto, o modelo de estimativa e a decodificação dos keypoints.

### Tracking
Usa informação de frames anteriores para estabilizar e acompanhar a pose no tempo.

### Mãos
Usa a pose corporal para localizar regiões das mãos e classificar apenas os estados necessários nesta fase: `OPEN`, `CLOSED` e `UNKNOWN`.

### Visualização
Desenha o resultado sem alterar o estado do tracking ou executar inferência.

## Regra da Fase 1

Evitar integrar uma solução completa de pose pronta. PyTorch, OpenCV e NumPy são infraestrutura; o objetivo continua sendo compreender e construir o pipeline de estimativa e tracking.
