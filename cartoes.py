import cv2
import numpy as np
import os
import csv


def processar_imagem(caminho_imagem: str) -> np.ndarray:
    imagem = cv2.imread(caminho_imagem)
    imagem_debug = imagem.copy()
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    _, imagem_binaria = cv2.threshold(imagem_suavizada, 60, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(imagem_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    triangulos = []
    for contorno in contornos:
        epsilon = 0.03 * cv2.arcLength(contorno, True)
        aproximacao = cv2.approxPolyDP(contorno, epsilon, True)
        if len(aproximacao) == 3:
            area = cv2.contourArea(aproximacao)
            if area > 500:
                triangulos.append(aproximacao)
                cv2.drawContours(imagem_debug, [aproximacao], -1, (0, 255, 0), 2)

    centros_triangulos = []
    for triangulo in triangulos:
        momentos = cv2.moments(triangulo)
        if momentos["m00"] != 0:
            cx = int(momentos["m10"] / momentos["m00"])
            cy = int(momentos["m01"] / momentos["m00"])
            centros_triangulos.append((cx, cy))

    if len(centros_triangulos) < 4:
        raise ValueError(f"A imagem '{caminho_imagem}' possui apenas {len(centros_triangulos)} triângulo(s) detectado(s).")

    centros_triangulos.sort(key=lambda p: p[1])
    dois_superiores = sorted(centros_triangulos[:2], key=lambda p: p[0])
    dois_inferiores = sorted(centros_triangulos[2:], key=lambda p: p[0], reverse=True)
    pontos_ordenados = np.array([dois_superiores[0], dois_superiores[1], dois_inferiores[0], dois_inferiores[1]], dtype="float32")

    largura = max(int(np.linalg.norm(pontos_ordenados[0] - pontos_ordenados[1])),
                  int(np.linalg.norm(pontos_ordenados[2] - pontos_ordenados[3])))
    altura = max(int(np.linalg.norm(pontos_ordenados[0] - pontos_ordenados[3])),
                 int(np.linalg.norm(pontos_ordenados[1] - pontos_ordenados[2])))

    pontos_destino = np.array([[0, 0], [largura - 1, 0], [largura - 1, altura - 1], [0, altura - 1]], dtype="float32")
    matriz_transformacao = cv2.getPerspectiveTransform(pontos_ordenados, pontos_destino)
    imagem_transformada = cv2.warpPerspective(imagem, matriz_transformacao, (largura, altura))

    pixels_corte_inferior = 200
    pixels_corte_superior = 150
    altura_final = altura - pixels_corte_inferior
    imagem_transformada = imagem_transformada[pixels_corte_superior:altura_final, 0:largura]

    return imagem_transformada


def recortar_colunas(imagem: np.ndarray) -> list:
    altura, largura = imagem.shape[:2]
    largura_coluna = largura // 3
    colunas_otimizadas = []

    for i in range(3):
        inicio_coluna = i * largura_coluna
        fim_coluna = (i + 1) * largura_coluna if i < 2 else largura
        coluna = imagem[:, inicio_coluna:fim_coluna]
        coluna_cinza = cv2.cvtColor(coluna, cv2.COLOR_BGR2GRAY)
        coluna_bin = cv2.adaptiveThreshold(coluna_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contornos, _ = cv2.findContours(coluna_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            x_min = largura_coluna
            y_min = altura
            x_max = 0
            y_max = 0

            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(contorno)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)

            margem = -7
            y_min = max(0, y_min - margem)
            y_max = min(altura, y_max + margem)
            x_min = max(0, x_min - margem)
            x_max = min(largura_coluna, x_max + margem)
            coluna_otimizada = coluna[y_min:y_max, x_min:x_max]
            colunas_otimizadas.append(coluna_otimizada)

    return colunas_otimizadas


def separar_questoes(coluna: np.ndarray, salvar_imagens: bool = False) -> list:
    if salvar_imagens:
        pasta_questoes = "questoes_debug"
        pasta_opcoes = os.path.join(pasta_questoes, "opcoes")
        os.makedirs(pasta_questoes, exist_ok=True)
        os.makedirs(pasta_opcoes, exist_ok=True)

    coluna_cinza = cv2.cvtColor(coluna, cv2.COLOR_BGR2GRAY)
    altura, largura = coluna_cinza.shape
    altura_questao = altura // 20
    respostas = []

    for i in range(20):
        y_inicio = i * altura_questao
        y_fim = (i + 1) * altura_questao
        questao = coluna_cinza[y_inicio:y_fim, :]

        if salvar_imagens:
            questao_debug = cv2.cvtColor(questao, cv2.COLOR_GRAY2BGR)
            largura_parte = largura // 6
            for j in range(1, 6):
                x = j * largura_parte
                cv2.line(questao_debug, (x, 0), (x, questao.shape[0]), (0, 0, 255), 1)
            labels = ['#', 'A', 'B', 'C', 'D', 'E']
            for j, label in enumerate(labels):
                x = (j * largura_parte) + (largura_parte // 2) - 10
                cv2.putText(questao_debug, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            caminho_debug = os.path.join(pasta_opcoes, f"questao_{i+1}_opcoes.jpg")
            cv2.imwrite(caminho_debug, questao_debug)

        largura_parte = largura // 6
        opcoes = []

        for j in range(1, 6):
            x_inicio = j * largura_parte
            x_fim = (j + 1) * largura_parte
            opcao = questao[:, x_inicio:x_fim]
            _, opcao_bin = cv2.threshold(opcao, 127, 255, cv2.THRESH_BINARY_INV)
            pixels_pretos = cv2.countNonZero(opcao_bin)
            opcoes.append(pixels_pretos)

        max_pixels = max(opcoes)
        if max_pixels > 600:
            resposta = chr(65 + opcoes.index(max_pixels))
        else:
            resposta = '-'

        respostas.append(resposta)

    return respostas


def main():
    pasta_imagens = "img_anonimizado"
    arquivos = sorted([f for f in os.listdir(pasta_imagens) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    if not arquivos:
        print("Nenhuma imagem encontrada na pasta.")
        return

    dados_formatados = []

    for numero_candidato, nome_arquivo in enumerate(arquivos, 1):
        caminho_imagem = os.path.join(pasta_imagens, nome_arquivo)
        print(f"\nProcessando: {nome_arquivo}")

        try:
            imagem_processada = processar_imagem(caminho_imagem)
            colunas = recortar_colunas(imagem_processada)
            respostas_cartao = []

            for indice, coluna in enumerate(colunas, 1):
                respostas = separar_questoes(coluna, salvar_imagens=(indice == 1))
                respostas_cartao.extend(respostas)

            if all(resposta == '-' for resposta in respostas_cartao):
                print(f" Cartão {numero_candidato} ignorado - todas as questões em branco: {nome_arquivo}")
                continue

            for i, resposta in enumerate(respostas_cartao, start=1):
                dados_formatados.append([i, resposta, numero_candidato])

            print(f"\nRespostas (candidato {numero_candidato}):")
            for num, resp in enumerate(respostas_cartao, 1):
                print(f"Questão {num}: {resp}")

        except Exception as e:
            print(f" Erro ao processar candidato {numero_candidato}: {e}")
            continue

    if dados_formatados:
        with open("respostas.csv", "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv, delimiter=',')
            writer.writerow(["questao", "resposta", "candidato"])
            writer.writerows(dados_formatados)
        candidatos_processados = len(set(row[2] for row in dados_formatados))
        print(f"\nRespostas salvas em 'respostas.csv'!")
        print(f"Total de cartões válidos processados: {candidatos_processados}")
    else:
        print("\nNenhum cartão válido foi processado!")


if __name__ == "__main__":
    main()
