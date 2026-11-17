import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

def extrair_id_paciente(caminho_ficheiro):
    nome_ficheiro = os.path.basename(caminho_ficheiro)
    partes = nome_ficheiro.split('_')
    if len(partes) >= 2:
        id_paciente = f"{partes[0]}_{partes[1]}"
        return id_paciente
    return nome_ficheiro

def simular_laudo_medico():
    print("--- [Fase 1] Preparando o Consultório da IA ---")
    
    try:
        modelo = tf.keras.models.load_model('melhor_ep_3c_27_10.keras') 
        print("Modelo carregado com sucesso!")
    except:
        print("Erro: Não foi possível carregar o modelo.")
        return

    # MUDANÇA AQUI: Apontando para a pasta Validacao
    dataset_teste = tf.keras.utils.image_dataset_from_directory(
        'Validacao',
        image_size=(128, 128),
        batch_size=32,
        color_mode='grayscale',
        label_mode='int',
        shuffle=False 
    )
    
    nomes_classes = dataset_teste.class_names
    caminhos_ficheiros = dataset_teste.file_paths
    
    print("\n--- [Fase 2] Realizando Exames (Avaliando Fatias) ---")
    previsoes_fatias_prob = modelo.predict(dataset_teste, verbose=1)
    previsoes_fatias = np.argmax(previsoes_fatias_prob, axis=1)
    gabaritos_fatias = np.concatenate([y for x, y in dataset_teste], axis=0)

    print("\n--- [Fase 3] Votação por Maioria (O Laudo Final) ---")
    
    pacientes_dados = {}
    
    for i in range(len(caminhos_ficheiros)):
        id_paciente = extrair_id_paciente(caminhos_ficheiros[i])
        chute_ia = previsoes_fatias[i]
        doenca_real = gabaritos_fatias[i]
        
        if id_paciente not in pacientes_dados:
            pacientes_dados[id_paciente] = {
                'votos_ia': [],
                'doenca_real': doenca_real 
            }
            
        pacientes_dados[id_paciente]['votos_ia'].append(chute_ia)

    gabaritos_pacientes = []
    previsoes_pacientes = []
    
    total_pacientes = len(pacientes_dados)
    pacientes_certos = 0
    
    for id_paciente, dados in pacientes_dados.items():
        votos = dados['votos_ia']
        doenca_real = dados['doenca_real']
        
        voto_maioria = Counter(votos).most_common(1)[0][0]
        
        gabaritos_pacientes.append(doenca_real)
        previsoes_pacientes.append(voto_maioria)
        
        if voto_maioria == doenca_real:
            pacientes_certos += 1

    acuracia_paciente = (pacientes_certos / total_pacientes) * 100
    
    print("\n==================================================")
    print(" 🏥 RESULTADO DO MUNDO REAL (VOTAÇÃO POR PACIENTE) 🏥")
    print("==================================================")
    print(f"Total de Fatias Avaliadas: {len(caminhos_ficheiros)}")
    print(f"Total de Pacientes Únicos: {total_pacientes}")
    print(f"Pacientes Diagnosticados Corretamente: {pacientes_certos}")
    print(f"-> ACURÁCIA FINAL (Por Paciente): {acuracia_paciente:.2f}% <-")
    
    # ==============================================================
    # A SOLUÇÃO DO ERRO: Forçar o Scikit-learn a ler as 4 classes
    # ==============================================================
    labels_idx = list(range(len(nomes_classes))) # Cria a lista [0, 1, 2, 3]
    
    print("\nRELATÓRIO DE CLASSIFICAÇÃO (PACIENTES):")
    print(classification_report(
        gabaritos_pacientes, 
        previsoes_pacientes, 
        target_names=nomes_classes, 
        labels=labels_idx, 
        zero_division=0 # Evita outro aviso chato se alguma classe estiver zerada
    ))

    print("\nMATRIZ DE CONFUSÃO (PACIENTES):")
    print(confusion_matrix(gabaritos_pacientes, previsoes_pacientes, labels=labels_idx))

if __name__ == "__main__":
    simular_laudo_medico()