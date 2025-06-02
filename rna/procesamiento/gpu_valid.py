import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("Dispositivos GPU físicos encontrados:", gpus)

# Imprimir nombre y capacidad de cada GPU
for gpu in gpus:
    details = tf.config.experimental.get_device_details(gpu)
    print(f"  → {gpu.name}, detalles: {details}")
    
if not gpus:
    print("No se encontró GPU. Revisa tu instalación de drivers/CUDA/cuDNN y que tengas instalado tensorflow-gpu.")
