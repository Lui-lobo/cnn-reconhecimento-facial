# Fotos de teste das emoções

160 fotos públicas de rostos com a emoção já rotulada, **20 de cada uma das 8 emoções** do FER+ (uma pasta por
emoção). Servem só para **testar e documentar** o modelo pronto de emoções: mostram o FER+ respondendo a cada emoção,
inclusive as que o grupo não consegue encenar na webcam. Para rodar: `python -m src.teste_emocao`.

- **Origem das fotos:** conjunto de teste (`PrivateTest`, 3589 fotos) do **FER2013**, criado para o desafio
  *Challenges in Representation Learning: Facial Expression Recognition* (Goodfellow et al., ICML 2013). São rostos
  já recortados, 48×48, em tons de cinza, reunidos de buscas de imagens na internet (por isso alguns têm marca d'água).
- **Origem dos rótulos:** **FER+** (Microsoft, [microsoft/FERPlus](https://github.com/microsoft/FERPlus), arquivo
  `fer2013new.csv`), em que 10 pessoas votaram na emoção de cada foto.
- **Por que o conjunto de teste:** o FER+ foi treinado com a parte de treino do FER2013; estas fotos ficaram de fora,
  então ele nunca as viu.
- **Seleção:** ficamos com as fotos em que uma emoção teve mais votos que todas as outras (sem empate) e pelo menos
  4 dos 10 votos, e sorteamos 20 de cada emoção (semente 42). O sorteio **não olhou a resposta do modelo**: há fotos
  que ele acerta e fotos que ele erra. “Com nojo” e “com desprezo” são raras: o conjunto de teste inteiro só tem 22 e
  23 fotos assim.
- **Arquivos:** as imagens não foram alteradas; o nome (`fer0032254.png`...) é o mesmo da lista do FER+.
- **Uso:** somente teste. Nenhuma destas fotos entra no treino da nossa CNN, e o FER+ não é retreinado.

As figuras do teste (`exemplos.png`, `matriz.png` e uma miniatura por emoção) são gravadas pelo
`python -m src.teste_emocao` em `web/img/emocao/` e aparecem no README e na seção 12 do site.
