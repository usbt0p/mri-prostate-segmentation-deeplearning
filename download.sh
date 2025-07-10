#!/bin/bash

BASE_URL="https://zenodo.org/records/6624726/files"
DEST_DIR="."
FOLD_COUNT=4  # hasta fold 4

for i in $(seq 0 "$FOLD_COUNT"); do
    ZIP_NAME="picai_public_images_fold${i}.zip"
    DOWNLOAD_URL="${BASE_URL}/${ZIP_NAME}?download=1"
    TARGET_DIR="picai_images_fold${i}"

    echo "Descargando fold $i..."
    curl -L -o "${DEST_DIR}/${ZIP_NAME}" "$DOWNLOAD_URL"
    if [ $? -ne 0 ]; then
        echo "Error al descargar ${ZIP_NAME}, abortando."
        exit 1
    fi

    echo "Creando directorio ${TARGET_DIR} y descomprimiendo..."
    mkdir -p "${DEST_DIR}/${TARGET_DIR}"
    unzip -q "${DEST_DIR}/${ZIP_NAME}" -d "${DEST_DIR}/${TARGET_DIR}"
    if [ $? -ne 0 ]; then
        echo "Error al descomprimir ${ZIP_NAME}, abortando."
        exit 1
    fi

    echo "Fold $i descargado y descomprimido en ${TARGET_DIR}."
done

echo "Todo descargado y descomprimido correctamente."
