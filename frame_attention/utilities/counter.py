def count_ids_with_valid_sequences(file_path, min_length=30):
    # Citim ID-urile din fișier și le stocăm într-o listă
    with open(file_path, 'r') as file:
        ids = [line.strip() for line in file if line.strip().isdigit()]  # Extragere ID-uri numerice

    # Set pentru a păstra ID-urile unice cu secvențe valide
    unique_ids_with_valid_sequence = set()

    # Variabile pentru a monitoriza secvențele curente
    current_id = None
    current_count = 0

    for id in ids:
        if id == current_id:
            current_count += 1  # Continuă secvența
        else:
            # Verifică dacă secvența anterioară este validă
            if current_id is not None and current_count >= min_length:
                unique_ids_with_valid_sequence.add(current_id)
            # Resetează pentru o nouă secvență
            current_id = id
            current_count = 1

    # Verifică ultima secvență
    if current_id is not None and current_count >= min_length:
        unique_ids_with_valid_sequence.add(current_id)

    # Returnăm numărul de ID-uri unice cu secvențe valide
    return len(unique_ids_with_valid_sequence)

# Apelăm funcția și afișăm rezultatul
file_path = 'tracked_ids.txt'
unique_id_count = count_ids_with_valid_sequences(file_path)
print(f"Numărul de tomberoane gasit este: {unique_id_count}")
