import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def extrair_padroes_kdd():
    print("--- [Fase de Avaliação KDD] Iniciando Extração de Padrões ---")
    
    # 1. Carregar o modelo treinado (O "Conhecimento" já minerado)
    try:
        modelo = tf.keras.models.load_model('modelo_otimizado2.keras')
        print("Modelo carregado com sucesso!")
    except:
        print("Erro: Não foi possível encontrar o ficheiro 'modelo_oasis_otimiza.keras'. Tens de treinar e salvar o modelo primeiro.")
        return

    # 2. Carregar apenas os dados de Teste
    dataset_teste = tf.keras.utils.image_dataset_from_directory(
        'Validacao',
        image_size=(128, 128),
        batch_size=32,
        color_mode='grayscale',
        label_mode='int',
        shuffle=False # Não baralhar para podermos rastrear os erros
    )
    
    nomes_classes = dataset_teste.class_names

    # ==========================================
    # PADRÃO 1: ANÁLISE PROFUNDA DE ERROS
    # ==========================================
    print("\n[1] A extrair padrões de falha (Análise de Erros)...")
    
    imagens_erro = []
    gabaritos_erro = []
    chutes_erro = []
    
    for imagens, labels in dataset_teste:
        previsoes = modelo.predict(imagens, verbose=0)
        chutes = np.argmax(previsoes, axis=1)
        gabaritos = labels.numpy()
        
        # Encontrar onde o chute foi diferente do gabarito
        erros_idx = np.where(chutes != gabaritos)[0]
        
        for idx in erros_idx:
            imagens_erro.append(imagens[idx])
            gabaritos_erro.append(gabaritos[idx])
            chutes_erro.append(chutes[idx])
            
            # Guardar apenas os 9 primeiros erros para visualizar
            if len(imagens_erro) >= 9:
                break
        if len(imagens_erro) >= 9:
            break

    # Desenhar o padrão de erros
    plt.figure(figsize=(12, 10))
    plt.suptitle("Padrões de Erro: O que a IA está a confundir?", fontsize=16)
    for i in range(min(9, len(imagens_erro))):
        ax = plt.subplot(3, 3, i + 1)
        # Converter a imagem de volta para formato visualizável
        img_visual = imagens_erro[i].numpy().astype("uint8")
        plt.imshow(img_visual, cmap='gray')
        plt.title(f"Real: {nomes_classes[gabaritos_erro[i]]}\nChute: {nomes_classes[chutes_erro[i]]}", color='red')
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ==========================================
    # PADRÃO 2: MAPAS DE CARACTERÍSTICAS (O que a IA vê?)
    # ==========================================
    print("\n[2] A extrair Feature Maps (O que a rede aprendeu a ver?)...")
    
    # Pegar apenas uma imagem do dataset de teste para servir de cobaia
    for imagens, labels in dataset_teste.take(1):
        imagem_cobaia = imagens[0:1] # Pegar a primeira imagem
        classe_real = nomes_classes[labels[0].numpy()]
        break

    # Criar um "sub-modelo" que para na primeira camada convolucional
    # Queremos ver a saída exata após os filtros matemáticos
    camada_conv_nome = [layer.name for layer in modelo.layers if 'conv2d' in layer.name][0]
    sub_modelo = tf.keras.models.Model(
        inputs=modelo.inputs, 
        outputs=modelo.get_layer(camada_conv_nome).output
    )

    # Passar a imagem pelo sub-modelo
    feature_maps = sub_modelo.predict(imagem_cobaia, verbose=0)

    # O feature map tem muitos filtros (ex: 32). Vamos desenhar os primeiros 16.
    plt.figure(figsize=(12, 12))
    plt.suptitle(f"Radiografia da IA: Procurando atrofias (Classe: {classe_real})", fontsize=16)
    
    # Plotar a imagem original no centro (como referência)
    ax = plt.subplot(5, 4, 1)
    plt.imshow(imagem_cobaia[0].numpy().astype("uint8"), cmap='gray')
    plt.title("Imagem Original")
    plt.axis("off")

    # Plotar os filtros aprendidos
    for i in range(1, 16):
        ax = plt.subplot(5, 4, i + 4) # Começa na linha de baixo
        # Aceder ao filtro específico 'i'
        filtro_visual = feature_maps[0, :, :, i-1]
        plt.imshow(filtro_visual, cmap='viridis') # Viridis ajuda a destacar texturas
        plt.title(f"Filtro/Padrão {i}")
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()

    print("\nProcesso KDD concluído! Podes usar estas imagens no teu TCC para justificar as tuas conclusões.")

if __name__ == "__main__":
    extrair_padroes_kdd()