# Fase 1 — Base do tracking corporal

## Objetivo

Construir a primeira versão funcional do pipeline de tracking corporal 2D, entendendo cada etapa em vez de apenas integrar um tracker pronto.

## Escopo

- uma pessoa por vez;
- uma webcam RGB comum;
- execução em tempo real como objetivo final da fase;
- 17 keypoints corporais;
- esqueleto 2D sobre o vídeo;
- suavização temporal;
- mão aberta, fechada ou desconhecida.

## Fora do escopo inicial

- múltiplas pessoas;
- reconstrução corporal 3D;
- rastreamento detalhado de dedos;
- reconhecimento de gestos complexos;
- identificação de pessoas;
- integração com engines de jogos;
- otimização extrema para produção.

## Ordem planejada de desenvolvimento

1. Preparar ambiente Python e validar a webcam. ✅
2. Implementar captura e visualização básica de frames. ✅
3. Definir formalmente os keypoints e conexões do esqueleto. ✅
4. Preparar dataset e pipeline de dados para pose. ✅ base implementada
5. Implementar uma primeira CNN de estimativa de pose. ✅ arquitetura implementada
6. Treinar e avaliar a estimativa por frame. ← etapa prática atual
7. Decodificar e desenhar os keypoints no vídeo. ✅ infraestrutura implementada; depende dos pesos treinados
8. Adicionar tracking temporal e suavização.
9. Localizar regiões das mãos a partir da pose.
10. Implementar classificação `OPEN`, `CLOSED` e `UNKNOWN`.
11. Medir FPS, estabilidade, latência e falhas comuns.

## Pipeline de pose implementado

```text
COCO person keypoints
  -> recorte de uma pessoa
  -> imagem 256x256
  -> 17 heatmaps alvo 64x64
  -> PoseNet
  -> 17 heatmaps previstos
  -> decoder
  -> 17 keypoints normalizados
  -> renderer
```

Na inferência pela webcam, a primeira versão assume uma pessoa centralizada e ocupando boa parte do frame. Ainda não existe detecção multi-pessoa nem tracking temporal.

## Critério de conclusão

A fase estará concluída quando uma pessoa em frente à webcam puder ser representada por um esqueleto 2D razoavelmente estável, com os principais movimentos acompanhados ao longo do tempo e uma indicação simples do estado de cada mão.
