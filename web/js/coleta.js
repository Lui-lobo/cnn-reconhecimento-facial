// Coleta de fotos para o dataset pelo navegador.
// "Ver câmera" liga a webcam antes da captura e desenha o oval-guia, para a pessoa se posicionar com calma.
// "Iniciar captura" faz uma contagem regressiva de 3 s, tira N fotos (uma a cada 250 ms) e envia para /api/fotos,
// que salva em dataset/<nome>/. Depois atualiza a tabela do passo 2 e, se a opção estiver marcada, já começa o treino.
// Usa esperar, enviarJSON, capturarQuadro e mostrarStatus de ao_vivo.js, e atualizarDataset/iniciarTreino de treino.js.

// Posição do rosto: os MESMOS valores de FRACAO_ROSTO e ALTURA_ROSTO em src/config.py
// (o oval desenhado aqui é exatamente a região que o pré-processamento recorta e mantém).
const FRACAO_ROSTO = 0.6;
const ALTURA_ROSTO = 0.45;

const videoColeta = document.getElementById("coleta-video");
const guiaColeta = document.getElementById("coleta-guia");      // canvas por cima do vídeo, com o oval
const botaoColeta = document.getElementById("coleta-iniciar");
const botaoPrevia = document.getElementById("coleta-previa");
const statusColeta = document.getElementById("coleta-status");
const contagem = document.getElementById("coleta-contagem");

let streamColeta = null;    // câmera da coleta (ligada pela prévia ou pela captura)

function normalizarNome(texto) {
  // "Ana Clara" -> "ana_clara": minúsculas, sem acento e sem espaço (o nome vira o nome da pasta)
  return texto.trim().toLowerCase()
    .normalize("NFD").replace(/[̀-ͯ]/g, "")    // remove os acentos
    .replace(/\s+/g, "_");
}

function desenharOvalGuia() {
  // Escurece o vídeo fora do oval do rosto e contorna o oval. A conta é a mesma de caixa_do_rosto (Python):
  // quadrado com 60% do menor lado, centralizado na horizontal e com o centro a 45% da altura.
  const largura = videoColeta.videoWidth, altura = videoColeta.videoHeight;
  guiaColeta.width = largura;
  guiaColeta.height = altura;
  const ctx = guiaColeta.getContext("2d");
  const lado = Math.min(largura, altura) * FRACAO_ROSTO;
  const cx = largura / 2;
  const cy = Math.min(Math.max(altura * ALTURA_ROSTO, lado / 2), altura - lado / 2);   // o quadrado não sai da tela
  const oval = () => { ctx.beginPath(); ctx.ellipse(cx, cy, 0.40 * lado, 0.48 * lado, 0, 0, 2 * Math.PI); };

  ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
  ctx.fillRect(0, 0, largura, altura);                 // escurece tudo...
  ctx.globalCompositeOperation = "destination-out";
  oval();
  ctx.fill();                                          // ...e "recorta" o oval, que fica com o vídeo normal
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = Math.max(2, largura / 160);
  ctx.setLineDash([12, 8]);
  oval();
  ctx.stroke();                                        // contorno tracejado do oval
}

async function ligarCameraColeta() {
  // Liga a webcam da coleta (se ainda não estiver ligada) e desenha o oval-guia por cima do vídeo
  if (streamColeta) return true;
  try {
    streamColeta = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  } catch (erro) {
    mostrarStatus(statusColeta, "Não foi possível abrir a webcam: " + erro.message, true);
    return false;
  }
  videoColeta.srcObject = streamColeta;
  await videoColeta.play();
  desenharOvalGuia();
  botaoPrevia.textContent = "Desligar câmera";
  return true;
}

function desligarCameraColeta() {
  // Desliga a webcam da coleta e apaga o oval-guia
  if (streamColeta) streamColeta.getTracks().forEach((faixa) => faixa.stop());
  streamColeta = null;
  videoColeta.srcObject = null;
  guiaColeta.getContext("2d").clearRect(0, 0, guiaColeta.width, guiaColeta.height);
  botaoPrevia.textContent = "Ver câmera";
}

async function alternarPrevia() {
  // Botão "Ver câmera": liga a câmera para a pessoa se posicionar no oval antes de capturar (ou desliga)
  if (streamColeta) {
    desligarCameraColeta();
    mostrarStatus(statusColeta, "Câmera desligada.");
  } else if (await ligarCameraColeta()) {
    mostrarStatus(statusColeta, "Encaixe o rosto no oval tracejado e clique em “Iniciar captura” quando estiver pronto.");
  }
}

async function coletarFotos() {
  const nome = normalizarNome(document.getElementById("coleta-nome").value);
  const n = Number(document.getElementById("coleta-n").value);
  if (!/^[a-z0-9][a-z0-9_-]*$/.test(nome)) {
    mostrarStatus(statusColeta, "Digite um nome usando só letras, números, _ ou -.", true);
    return;
  }
  if (!(n >= 1 && n <= 400)) {
    mostrarStatus(statusColeta, "A quantidade de fotos deve ficar entre 1 e 400.", true);
    return;
  }

  // 1. Liga a webcam (se a prévia já estava ligada, usa a mesma câmera e a deixa ligada no fim)
  botaoColeta.disabled = true;
  botaoPrevia.disabled = true;
  const previaLigada = streamColeta !== null;
  if (!(await ligarCameraColeta())) {
    botaoColeta.disabled = false;
    botaoPrevia.disabled = false;
    return;
  }

  // 2. Contagem regressiva de 3 segundos para a pessoa se posicionar
  mostrarStatus(statusColeta, `${nome}: encaixe o rosto no oval.`);
  for (let s = 3; s > 0; s--) {
    contagem.textContent = s;
    await esperar(1000);
  }
  contagem.textContent = "";

  // 3. Tira n fotos, uma a cada 250 ms (guardadas como JPEG em texto). O oval é só um desenho na tela:
  //    a foto é copiada do vídeo, sem ele.
  const fotos = [];
  for (let i = 0; i < n; i++) {
    fotos.push(capturarQuadro(videoColeta, 0.9));
    mostrarStatus(statusColeta, `${nome}: foto ${i + 1} de ${n} (mexa levemente a cabeça, sem sair do oval)`);
    await esperar(250);
  }

  // 4. Desliga a câmera (se foi ligada só para a captura) e envia as fotos ao servidor
  if (!previaLigada) desligarCameraColeta();
  botaoPrevia.disabled = false;
  mostrarStatus(statusColeta, "Enviando as fotos para o servidor...");
  try {
    const r = await enviarJSON("/api/fotos", { nome, imagens: fotos });
    const treinarAgora = document.getElementById("coleta-treinar").checked;
    mostrarStatus(statusColeta, `${r.salvas} fotos salvas em ${r.pasta}/. ` + (treinarAgora
      ? "Começando o treino (passo 2)..."
      : "Quando terminar a coleta de todos, clique em “Treinar modelo” (passo 2)."));
    await atualizarDataset();
    if (treinarAgora) iniciarTreino();
  } catch (erro) {
    mostrarStatus(statusColeta, erro.message, true);
  }
  botaoColeta.disabled = false;
}

botaoPrevia.addEventListener("click", alternarPrevia);
botaoColeta.addEventListener("click", coletarFotos);
