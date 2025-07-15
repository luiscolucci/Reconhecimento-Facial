import face_recognition
import cv2
import os
import numpy as np

# --- 1. Carregar as imagens conhecidas e seus nomes ---
caminho_imagens_conhecidas = "imagens_conhecidas"
codificacoes_faciais_conhecidas = []
nomes_conhecidos = []

print("Carregando imagens conhecidas...")
for nome_arquivo in os.listdir(caminho_imagens_conhecidas):
    # Carrega a imagem
    caminho_completo = os.path.join(caminho_imagens_conhecidas, nome_arquivo)
    imagem = face_recognition.load_image_file(caminho_completo)

    # Gera a codificação facial
    # Acessamos o primeiro elemento, pois pode haver mais de um rosto na imagem
    codificacao = face_recognition.face_encodings(imagem)[0]

    # Adiciona a codificação e o nome (sem a extensão do arquivo) às listas
    codificacoes_faciais_conhecidas.append(codificacao)
    nomes_conhecidos.append(os.path.splitext(nome_arquivo)[0])

print("Imagens conhecidas carregadas com sucesso.")

# --- 2. Captura de vídeo da webcam ---
captura_video = cv2.VideoCapture(0) # 0 para a webcam padrão

# Inicializa variáveis
localizacoes_faciais = []
codificacoes_faciais = []
nomes_faciais = []
processar_este_quadro = True

while True:
    # Captura um único quadro de vídeo
    ret, quadro = captura_video.read()
    if not ret:
        break

    # Redimensiona o quadro para um processamento mais rápido
    quadro_pequeno = cv2.resize(quadro, (0, 0), fx=0.25, fy=0.25)

    # Converte a imagem de BGR (usado pelo OpenCV) para RGB (usado pelo face_recognition)
    quadro_rgb_pequeno = np.ascontiguousarray(quadro_pequeno[:, :, ::-1])


    # Processa apenas quadros alternados para economizar tempo
    if processar_este_quadro:
        # Encontra todos os rostos e suas codificações no quadro atual
        localizacoes_faciais = face_recognition.face_locations(quadro_rgb_pequeno)
        codificacoes_faciais = face_recognition.face_encodings(quadro_rgb_pequeno, localizacoes_faciais)

        nomes_faciais = []
        for codificacao_facial in codificacoes_faciais:
            # Compara o rosto encontrado com os rostos conhecidos
            correspondencias = face_recognition.compare_faces(codificacoes_faciais_conhecidas, codificacao_facial)
            nome = "Desconhecido"

            # Usa o rosto conhecido com a menor distância para o novo rosto
            distancias_faciais = face_recognition.face_distance(codificacoes_faciais_conhecidas, codificacao_facial)
            melhor_indice_correspondencia = np.argmin(distancias_faciais)
            if correspondencias[melhor_indice_correspondencia]:
                nome = nomes_conhecidos[melhor_indice_correspondencia]

            nomes_faciais.append(nome)

    processar_este_quadro = not processar_este_quadro

    # --- 3. Exibir os resultados ---
    for (topo, direita, baixo, esquerda), nome in zip(localizacoes_faciais, nomes_faciais):
        # Redimensiona as localizações de volta ao tamanho original da imagem
        topo *= 4
        direita *= 4
        baixo *= 4
        esquerda *= 4

        # Desenha um retângulo ao redor do rosto
        cv2.rectangle(quadro, (esquerda, topo), (direita, baixo), (0, 255, 0), 2)

        # Desenha uma legenda com o nome abaixo do rosto
        cv2.rectangle(quadro, (esquerda, baixo - 35), (direita, baixo), (0, 255, 0), cv2.FILLED)
        fonte = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(quadro, nome, (esquerda + 6, baixo - 6), fonte, 1.0, (255, 255, 255), 1)

    # Exibe a imagem resultante
    cv2.imshow('Reconhecimento Facial', quadro)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o controle da webcam e fecha as janelas
captura_video.release()
cv2.destroyAllWindows()