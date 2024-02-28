# dossier des images
image_path = "letters"

# version de la configuration
version = 10

# Taille des images qui serviront à entraîner le modèle
image_size = 50

# marge blanche ajoutée aux images
padding = 25

# Taille du dataset
N = 1250*10*8

# Epoques pur l'entraînement
epochs = 50

image_folder = f"data/{image_path}"
original_image_paths = [f"{image_folder}/{i:02d}.jpg" for i in range(1, 9)]

# /train_{version}_{N}.pt
train_folder = f"dataset/{image_path}"
train_path = f"{train_folder}/train_{version}_{N}.pt"
test_path = f"{train_folder}/test_{version}_{N}.pt"

parameters_folder = "parameters/" + image_path

result_id = f'{version}_{N}_{epochs}'

parameters_path = f'{parameters_folder}/{image_path}_{result_id}.pth'

result_folder = "result/" + image_path
loss_path = f'{result_folder}/loss_{result_id}.jpg'
confusion_path = f'{result_folder}/confusion_{result_id}.jpg'
