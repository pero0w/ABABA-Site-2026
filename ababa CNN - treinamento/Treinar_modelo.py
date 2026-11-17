import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

def treinar_cnn():
    Data = 'Data' 
    TAMANHO_IMG = (128, 128)
    TAMANHO_LOTE = 32

    print("--- [Fase 1] Carregando a esteira de imagens ---")
    
    dataset_treino = tf.keras.utils.image_dataset_from_directory(
        Data,
        seed=123,
        image_size=TAMANHO_IMG,
        batch_size=TAMANHO_LOTE,
        color_mode='grayscale',
        label_mode='int'
    )

    dataset_validacao = tf.keras.utils.image_dataset_from_directory(
        'Validacao', 
        seed=123,
        image_size=TAMANHO_IMG,
        batch_size=TAMANHO_LOTE,
        color_mode='grayscale',
        label_mode='int'
    )

    dataset_teste = tf.keras.utils.image_dataset_from_directory(
        'Teste', image_size=TAMANHO_IMG, batch_size=TAMANHO_LOTE,
        color_mode='grayscale', label_mode='int', shuffle=False
    )

    nomes_classes = dataset_treino.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    dataset_treino = dataset_treino.cache().prefetch(buffer_size=AUTOTUNE)
    dataset_validacao = dataset_validacao.cache().prefetch(buffer_size=AUTOTUNE)
    dataset_teste = dataset_teste.cache().prefetch(buffer_size=AUTOTUNE)

    print("--- Construindo a Arquitetura ---")
    model = models.Sequential([
        layers.RandomFlip("horizontal", input_shape=(128, 128, 1)),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])

    learning_rate = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, patience=3, min_lr=0.00001, verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    print("--- [Fase 3] Compilando o Modelo ---")

    otimizador_suave = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=otimizador_suave,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary() 

    print("\n--- [Fase 3.5] Calculando Punição para Classes Raras ---")
    y_treino_rotulos = np.concatenate([y for x, y in dataset_treino], axis=0)
    pesos_array = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_treino_rotulos), y=y_treino_rotulos
    )
    pesos_dit = dict(enumerate(pesos_array))

    print("\n--- [Fase 4] Iniciando Treinamento ---")
    hist = model.fit(
        dataset_treino,
        validation_data=dataset_validacao,
        callbacks=[learning_rate, early_stop],
        epochs=15,
        class_weight=pesos_dit
    )

    print("\n--- [Fase Final] Avaliação no Teste Cego ---")
    perda, acuracia = model.evaluate(dataset_teste)
    print(f"Teste Cego -> Acurácia: {acuracia * 100:.2f}% | Perda: {perda:.4f}")

    print("\n--- [Fase 5] Gerando Gráficos ---")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='Treino')
    plt.plot(hist.history['val_accuracy'], label='Validação')
    plt.title('Acurácia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='Treino')
    plt.plot(hist.history['val_loss'], label='Validação')
    plt.title('Perda (Loss)')
    plt.legend()
    plt.show()

    print("\n--- [Fase 6] Relatório Avançado e Matriz de Confusão ---")
    y_true = []
    y_pred = []
    for imagens, labels in dataset_teste:
        previsoes_lote = model.predict(imagens, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(previsoes_lote, axis=1))

    print(classification_report(y_true, y_pred, target_names=nomes_classes))
    print("\nMATRIZ DE CONFUSÃO:")
    print(confusion_matrix(y_true, y_pred))

    model.save('modelo_otimizado4.keras')
    print("Modelo salvo!")

if __name__ == "__main__":
    treinar_cnn()