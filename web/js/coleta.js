// Coleta de fotos para o dataset pelo navegador.
// Liga a webcam, faz uma contagem regressiva de 3 s, tira N fotos (uma a cada 250 ms) e envia para /api/fotos,
// que salva em dataset/<nome>/. Usa esperar, enviarJSON, capturarQuadro e mostrarStatus de ao_vivo.js.

const videoColeta = document.getElementById("coleta-video");
const botaoColeta = document.getElementById("coleta-iniciar");
const statusColeta = document.getElementById("coleta-status");
const contagem = document.getElementById("coleta-contagem");

function normalizarNome(texto) {
  // "Ana Clara" -> "ana_clara": minúsculas, sem acento e sem espaço (o nome vira o nome da pasta)
  return texto.trim().toLowerCase()
    .normalize("NFD").replace(/[̀-ͯ]/g, "")    // remove os acentos
    .replace(/\s+/g, "_");
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

  // 1. Liga a webcam
  botaoColeta.disabled = true;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  } catch (erro) {
    mostrarStatus(statusColeta, "Não foi possível abrir a webcam: " + erro.message, true);
    botaoColeta.disabled = false;
    return;
  }
  videoColeta.srcObject = stream;
  await videoColeta.play();

  // 2. Contagem regressiva de 3 segundos para a pessoa se posicionar
  mostrarStatus(statusColeta, `${nome}: centralize o rosto.`);
  for (let s = 3; s > 0; s--) {
    contagem.textContent = s;
    await esperar(1000);
  }
  contagem.textContent = "";

  // 3. Tira n fotos, uma a cada 250 ms (guardadas como JPEG em texto)
  const fotos = [];
  for (let i = 0; i < n; i++) {
    fotos.push(capturarQuadro(videoColeta, 0.9));
    mostrarStatus(statusColeta, `${nome}: foto ${i + 1} de ${n} (mexa levemente a cabeça)`);
    await esperar(250);
  }

  // 4. Desliga a câmera e envia as fotos ao servidor
  stream.getTracks().forEach((faixa) => faixa.stop());
  videoColeta.srcObject = null;
  mostrarStatus(statusColeta, "Enviando as fotos para o servidor...");
  try {
    const r = await enviarJSON("/api/fotos", { nome, imagens: fotos });
    mostrarStatus(statusColeta, `${r.salvas} fotos salvas em ${r.pasta}/. Quando terminar a coleta de todos, ` +
      "rode python -m src.treinar de novo para retreinar a rede.");
  } catch (erro) {
    mostrarStatus(statusColeta, erro.message, true);
  }
  botaoColeta.disabled = false;
}

botaoColeta.addEventListener("click", coletarFotos);
