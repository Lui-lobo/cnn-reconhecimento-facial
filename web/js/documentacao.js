// Documentação: preenche a página com o código real, as imagens e os números do último treino.
// - <pre data-codigo="modulo/funcao">  -> código Python lido de src/ pela rota /api/codigo
// - <pre data-js="arquivo.js#funcao">  -> uma função JavaScript lida do próprio arquivo do site
// - <img data-resultado="nome">        -> imagem gerada pelo treino em img/resultados/nome.png
// - tabelas com id                     -> montadas a partir de /api/resultados (resultados.json)

const porcento = (v) => (v * 100).toFixed(1).replace(".", ",") + "%";   // 0.953 -> "95,3%"
const numero = (v) => Number(v).toLocaleString("pt-BR");                 // 18496 -> "18.496"

// ---------- Blocos de código ----------

function escaparHTML(texto) {
  // Troca os caracteres especiais do HTML para o código aparecer como texto
  return texto.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function colorirComentarios(codigo, marcador) {
  // Destaca os comentários (# no Python, // no JavaScript) com outra cor, linha por linha
  const regra = marcador === "#" ? /(^|\s)(#.*)$/ : /(^|\s)(\/\/.*)$/;
  return escaparHTML(codigo).split("\n")
    .map((linha) => linha.replace(regra, '$1<span class="comentario">$2</span>'))
    .join("\n");
}

function mostrarCodigo(pre, titulo, codigo, marcador) {
  // Coloca o código no <pre> com um título em cima; os blocos "recolhidos" ficam dentro de um <details>
  pre.innerHTML = "<code>" + colorirComentarios(codigo, marcador) + "</code>";
  if (pre.classList.contains("recolhido")) {
    const caixa = document.createElement("details");
    caixa.className = "codigo-recolhido";
    caixa.innerHTML = "<summary></summary>";
    caixa.firstChild.textContent = "Mostrar o código: " + titulo;
    pre.replaceWith(caixa);
    caixa.append(pre);
  } else {
    const legenda = document.createElement("div");
    legenda.className = "codigo-titulo";
    legenda.textContent = titulo;
    pre.before(legenda);
  }
}

async function carregarCodigoPython(pre) {
  // Busca no servidor o código-fonte atual de uma função de src/ (Python) ou do servidor.js
  const [modulo, funcao] = pre.dataset.codigo.split("/");
  try {
    const resposta = await fetch(`/api/codigo/${modulo}/${funcao}`);
    const dados = await resposta.json();
    if (!resposta.ok) throw new Error(dados.erro);
    mostrarCodigo(pre, `${dados.arquivo} · ${funcao}()`, dados.codigo, dados.arquivo.endsWith(".py") ? "#" : "//");
  } catch (erro) {
    mostrarCodigo(pre, `${modulo} · ${funcao}()`, `Não foi possível carregar o código (${erro.message}).`, "#");
  }
}

async function carregarCodigoJS(pre, arquivos) {
  // Recorta uma função do arquivo JavaScript: do "function nome(" até a primeira "}" no começo de uma linha
  const [arquivo, funcao] = pre.dataset.js.split("#");
  if (!arquivos[arquivo]) arquivos[arquivo] = fetch(`js/${arquivo}`).then((r) => r.text());   // baixa cada arquivo uma vez
  const texto = (await arquivos[arquivo]).replace(/\r\n/g, "\n");
  const inicio = texto.search(new RegExp(`^(async )?function ${funcao}\\(`, "m"));
  const fim = texto.indexOf("\n}", inicio);
  const codigo = inicio >= 0 ? texto.slice(inicio, fim + 2) : `// Função ${funcao} não encontrada.`;
  mostrarCodigo(pre, `web/js/${arquivo} · ${funcao}()`, codigo, "//");
}

// ---------- Imagens geradas pelo treino ----------

function carregarImagens(versao) {
  // Aponta cada <img data-resultado> para o PNG gerado; "?v=" força o navegador a buscar a versão nova após retreinar.
  // Se a imagem ainda não existe, ela fica escondida e um aviso aparece no lugar (até o próximo treino).
  document.querySelectorAll("img[data-resultado]").forEach((img) => {
    let aviso = img.nextElementSibling?.classList.contains("sem-imagem") ? img.nextElementSibling : null;
    img.onload = () => {
      img.hidden = false;
      if (aviso) aviso.remove();
    };
    img.onerror = () => {
      img.hidden = true;
      if (!aviso) {
        aviso = document.createElement("p");
        aviso.className = "sem-imagem";
        aviso.textContent = `Imagem “${img.dataset.resultado}.png” ainda não gerada: treine o modelo.`;
        img.after(aviso);
      }
    };
    img.src = `img/resultados/${img.dataset.resultado}.png` + (versao ? `?v=${encodeURIComponent(versao)}` : "");
  });
}

// ---------- Tabelas montadas com os resultados ----------

function tabela(cabecalho, linhas, classeUltimaLinha) {
  // Monta o HTML de uma tabela simples a partir de listas de textos
  const th = cabecalho.map((c) => `<th>${c}</th>`).join("");
  const trs = linhas.map((linha, i) => {
    const classe = classeUltimaLinha && i === linhas.length - 1 ? ` class="${classeUltimaLinha}"` : "";
    return `<tr${classe}>` + linha.map((c) => `<td>${c}</td>`).join("") + "</tr>";
  }).join("");
  return `<table><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table>`;
}

function blocosDeNumeros(itens) {
  // Monta os "cartões" com um número grande e uma legenda
  return itens.map(([valor, legenda]) => `<div class="cartao"><strong>${valor}</strong><span>${legenda}</span></div>`).join("");
}

function preencherResultados(r) {
  const nomes = r.classes.map(escaparHTML);
  const m = r.metricas;

  // Visão geral
  document.getElementById("resumo-geral").innerHTML = blocosDeNumeros([
    [r.classes.length, "integrantes"],
    [numero(r.totais.fotos), "fotos no dataset"],
    [porcento(r.acuracias.teste), "acurácia no teste"],
    [numero(r.parametros_total), "parâmetros na rede"],
  ]);

  // Divisão: fotos por integrante em cada conjunto
  const conj = r.fotos_por_conjunto;
  const linhasDivisao = r.classes.map((nome, i) =>
    [nomes[i], conj.treino[nome], conj.validacao[nome], conj.teste[nome]]);
  linhasDivisao.push(["<strong>total</strong>", r.totais.treino, r.totais.validacao, r.totais.teste]);
  document.getElementById("tabela-divisao").innerHTML =
    tabela(["Integrante", "Treino", "Validação", "Teste"], linhasDivisao, "total") +
    `<p class="nota">Com o aumento de dados, o treino passa de ${numero(r.totais.treino)} para ` +
    `${numero(r.totais.treino_com_aumento)} imagens (originais + 3 cópias modificadas).</p>`;

  // Arquitetura: camada a camada, como no model.summary()
  const linhasCamadas = r.camadas.map((c) =>
    [`<code>${c.nome}</code>`, c.tipo, "(" + c.saida.join(", ") + ")", numero(c.parametros)]);
  linhasCamadas.push(["<strong>total</strong>", "", "", `<strong>${numero(r.parametros_total)}</strong>`]);
  document.getElementById("tabela-camadas").innerHTML =
    tabela(["Camada", "Tipo", "Formato da saída", "Parâmetros"], linhasCamadas, "total");

  // Treinamento
  const ultima = r.melhor_epoca - 1;
  document.getElementById("info-treino").innerHTML = blocosDeNumeros([
    [r.epocas_treinadas, "épocas treinadas"],
    [r.melhor_epoca, "melhor época (pesos usados)"],
    [r.historico.val_loss[ultima].toFixed(3).replace(".", ","), "erro na validação"],
    [porcento(r.historico.val_accuracy[ultima]), "acurácia na validação"],
  ]);

  // Matriz de confusão em HTML: a cor de cada célula é a porcentagem da linha
  let html = '<div class="tabela-rolagem"><table class="matriz"><thead><tr><th class="canto">real ↓ / previsto →</th>' +
    nomes.map((n) => `<th>${n}</th>`).join("") + "</tr></thead><tbody>";
  r.matriz_confusao.forEach((linha, i) => {
    const totalLinha = Math.max(linha.reduce((a, b) => a + b, 0), 1);
    html += `<tr><th>${nomes[i]}</th>`;
    linha.forEach((valor, j) => {
      const p = valor / totalLinha;
      const classe = (i === j ? "diagonal" : "") + (p > 0.5 ? " forte" : "");
      html += `<td class="${classe}" style="--p: ${(p * 100).toFixed(0)}%">${valor}<small>${(p * 100).toFixed(0)}%</small></td>`;
    });
    html += "</tr>";
  });
  document.getElementById("matriz-html").innerHTML = html + "</tbody></table></div>";

  // Tabela de métricas por integrante + média macro
  const media = (lista) => lista.reduce((a, b) => a + b, 0) / lista.length;
  const linhasMetricas = r.classes.map((_, i) => [nomes[i], m.VP[i], m.FP[i], m.FN[i], m.VN[i],
    porcento(m.precisao[i]), porcento(m.recall[i]), porcento(m.f1[i]), porcento(m.especificidade[i])]);
  linhasMetricas.push(["<strong>média macro</strong>", "", "", "", "", porcento(media(m.precisao)),
    porcento(media(m.recall)), porcento(media(m.f1)), porcento(media(m.especificidade))]);
  document.getElementById("tabela-metricas").innerHTML =
    tabela(["Integrante", "VP", "FP", "FN", "VN", "Precisão", "Recall", "F1", "Especificidade"], linhasMetricas, "total") +
    `<p class="nota">Acurácia no teste: <strong>${porcento(m.acuracia)}</strong> (${r.totais.teste} fotos). ` +
    `Resultados do treino de ${escaparHTML(r.gerado_em)}.</p>`;

  // Acurácia nos três conjuntos
  document.getElementById("acuracias").innerHTML = blocosDeNumeros([
    [porcento(r.acuracias.treino), `treino (${numero(r.totais.treino)} fotos)`],
    [porcento(r.acuracias.validacao), `validação (${numero(r.totais.validacao)} fotos)`],
    [porcento(r.acuracias.teste), `teste (${numero(r.totais.teste)} fotos)`],
  ]);
}

async function carregarResultados() {
  // Busca o resultados.json; se o modelo ainda não foi treinado, avisa no topo da documentação e nas tabelas
  try {
    const resposta = await fetch("/api/resultados");
    const r = await resposta.json();
    if (!resposta.ok) throw new Error(r.erro);
    preencherResultados(r);
    document.getElementById("aviso-resultados").hidden = true;
    carregarImagens(r.gerado_em);
  } catch (erro) {
    const aviso = document.getElementById("aviso-resultados");
    aviso.textContent = erro.message;
    aviso.hidden = false;
    document.querySelectorAll(".dinamico").forEach((el) => {
      el.innerHTML = '<p class="sem-imagem">Disponível depois do primeiro treino.</p>';
    });
    carregarImagens(null);
  }
}

// ---------- Índice: destaca a seção que está na tela ----------

function acompanharIndice() {
  const links = document.querySelectorAll(".indice a[href^='#']");
  const observador = new IntersectionObserver((entradas) => {
    entradas.forEach((entrada) => {
      if (!entrada.isIntersecting) return;
      links.forEach((a) => a.classList.toggle("atual", a.getAttribute("href") === "#" + entrada.target.id));
    });
  }, { rootMargin: "-20% 0px -70% 0px" });
  document.querySelectorAll(".secao").forEach((secao) => observador.observe(secao));
}

// ---------- Início ----------

const arquivosJS = {};
document.querySelectorAll("pre[data-codigo]").forEach(carregarCodigoPython);
document.querySelectorAll("pre[data-js]").forEach((pre) => carregarCodigoJS(pre, arquivosJS));
carregarResultados();
acompanharIndice();
