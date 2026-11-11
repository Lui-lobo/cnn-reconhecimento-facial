# Rostos de fora do grupo

1500 fotos públicas de rostos, **cada uma de uma pessoa diferente**, usadas no treino como exemplos de
“gente que não é do grupo”. Elas não viram uma classe: para estas fotos, a rede é cobrada a dividir a resposta por
igual entre as classes, e assim aprende a só ter confiança alta para os integrantes (ver `src/desconhecidos.py`).

- **Origem:** *Labeled Faces in the Wild* (LFW), versão *funneled*, da Universidade de Massachusetts Amherst:
  fotos de pessoas públicas publicadas em notícias, reunidas para pesquisa em reconhecimento facial.
  Huang, Ramesh, Berg e Learned-Miller, *Labeled Faces in the Wild: A Database for Studying Face Recognition in
  Unconstrained Environments*, relatório técnico 07-49, 2007.
- **Seleção:** das 5749 pessoas do conjunto, sorteamos 1500 (semente 42) e ficamos com a primeira foto de cada uma.
  Os arquivos foram só renomeados (`fora_0000.jpg`...); as imagens (250×250) não foram alteradas.
- **Uso:** somente como dado de treino e de teste. Nenhum modelo pronto é usado: as fotos passam pelo mesmo
  pré-processamento das fotos dos integrantes e a CNN continua sendo treinada do zero.

Para reforçar com pessoas fotografadas na própria webcam (amigos, familiares), coletem fotos pelo site com o nome
`desconhecido`: elas vão para `dataset/desconhecido/` e entram no treino junto com estas.
