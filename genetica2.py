import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
# Funcao para gerar e salvar pontos
def generate_and_save_points(num_points=8, distribution_type=1):
    if distribution_type == 1:  # Pontos uniformemente distribuidos
        points = np.random.rand(num_points, 2) * 100
        folder_path = "distribuida"
    elif distribution_type == 2:  # Círculo de pontos
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        radius = 50
        center = (50, 50)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        points = np.column_stack((x, y))
        folder_path = "circular"
    else:
        raise ValueError("Tipo de distribuição inválido.")
    
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(os.path.join(folder_path, "points.npy"), points)
    print(f"Pontos gerados e salvos em {folder_path}/points.npy.")
    return points


def load_points(folder_name=""):
    points = np.load(os.path.join(folder_name, "points.npy"))
    print(f"Pontos carregados de {folder_name}/points.npy.")
    return points

# Funções do Algoritmo Genético
def total_distance(route, points):
    return sum(np.linalg.norm(points[route[i]] - points[route[i - 1]]) for i in range(len(route)))

def fitness(route, points):
    return 1 / total_distance(route, points)

def tournament_selection(population, points, tournament_size=5):
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda route: fitness(route, points), reverse=True)
    return selected[0]

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * len(parent1)
    child[start:end] = parent1[start:end]
    pos = end
    for gene in parent2:
        if gene not in child:
            if pos >= len(parent1):
                pos = 0
            child[pos] = gene
            pos += 1
    return child

def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm(points, pop_size, num_generations, mutation_rate, tournament_size):
    population = [random.sample(range(len(points)), len(points)) for _ in range(pop_size)]
    best_route = min(population, key=lambda route: total_distance(route, points))
    best_distance = total_distance(best_route, points)
    progress = [best_distance]

    for generation in range(num_generations):
        new_population = []
        for _ in range(pop_size):
            parent1 = tournament_selection(population, points, tournament_size)
            parent2 = tournament_selection(population, points, tournament_size)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        current_best_route = min(population, key=lambda route: total_distance(route, points))
        current_best_distance = total_distance(current_best_route, points)

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = current_best_route

        progress.append(best_distance)

    return best_route, best_distance, progress



# Funcao para salvar graficos com nome incremental e mais alguma infos
def plot_and_save_results(points, best_route, progress, folder_name="", file_prefix="result", execution_time=0.0):
    # Calcula a distancia total do melhor caminho
    best_distance = sum(np.linalg.norm(points[best_route[i]] - points[best_route[i - 1]]) for i in range(len(best_route)))

    plt.figure(figsize=(10, 6))

    # Melhor rota encontrada
    plt.subplot(1, 2, 1)
    route_points = np.array([points[i] for i in best_route] + [points[best_route[0]]])
    plt.plot(route_points[:, 0], route_points[:, 1], marker='o')
    plt.title("Melhor Rota Encontrada")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Progresso do algoritmo genetico
    plt.subplot(1, 2, 2)
    plt.plot(progress)
    plt.title("Progresso do Algoritmo Genético")
    plt.xlabel("Geração")
    plt.ylabel("Distância")

    plt.tight_layout()

    # Informações de tempo de execução e distancia total abaixo dos graficos
    info_text = f"Tempo de Execução: {execution_time:.8f} segundos\nTamanho do Caminho: {best_distance:.2f}"
    plt.figtext(0.5, -0.05, info_text, ha="center", fontsize=10)

    # Salva os graficos com nome incremental
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Encontra o proximo nome de arquivo disponivel
    file_index = 1
    while os.path.exists(os.path.join(folder_name, f"{file_prefix}_plots{file_index}.png")):
        file_index += 1
    file_path = os.path.join(folder_name, f"{file_prefix}_plots{file_index}.png")

    plt.savefig(file_path, bbox_inches="tight")
    plt.show()
    print(f"Gráficos salvos em {file_path}")


def main():
    option = input("Digite '1' para gerar novos pontos, '2' para carregar pontos salvos ou '3' para gerar ambos os tipos de pontos:\n -")
    if option == '1':
        num_points = int(input("Número de pontos:\n -"))
        distribution_type = int(input("Escolha a distribuição dos pontos:\n '1' para distribuição uniforme ou '2' para circular:\n -"))
        points = generate_and_save_points(num_points=num_points, distribution_type=distribution_type)

    elif option == '2':
        distribution_choice = int(input("Escolha o tipo de pontos para carregar:\n '1' para distribuído, '2' para circular ou '3' para ambos:\n -"))
        if distribution_choice == 1:
            points = load_points("distribuida")
        elif distribution_choice == 2:
            points = load_points("circular")
        elif distribution_choice == 3:
            points_distribuida = load_points("distribuida")
            points_circular = load_points("circular")
        else:
            print("Opção inválida.")
            return

    elif option == '3':
        num_points = int(input("Número de pontos:\n -"))
        # Gera e salva pontos para ambas as distribuicoes
        points_distribuida = generate_and_save_points(num_points=num_points, distribution_type=1)
        points_circular = generate_and_save_points(num_points=num_points, distribution_type=2)
    else:
        print("Opção inválida.")
        return

    # Solicita o tamanho da populacao ao usuario
    pop_size = int(input("Tamanho da população:\n -"))

    # Solicita parametros do Algoritmo Genetico
    num_generations = int(input("Número de gerações:\n -"))
    mutation_rate = float(input("Taxa de mutação (ex: 0.2):\n -"))
    tournament_size = int(input("Tamanho do torneio (ex: 5):\n -"))

    # Executa o algoritmo genetico e salva resultados para cada tipo de pontos
    if option == '3' or (option == '2' and distribution_choice == 3):
        # Executa para pontos distribuidos
        print("\nExecutando para pontos distribuídos...")
        start_time = time.time()  # Inicia a medicao do tempo
        best_route, best_distance, progress = genetic_algorithm(points_distribuida, pop_size, num_generations, mutation_rate, tournament_size)
        execution_time_distribuida = time.time() - start_time  # Finaliza a medicao do tempo
        plot_and_save_results(points_distribuida, best_route, progress, folder_name="distribuida", file_prefix="distribuida", execution_time=execution_time_distribuida)

        # Executa para pontos em circulo
        print("\nExecutando para pontos em círculo...")
        start_time = time.time()  # Inicia a medicao do tempo
        best_route, best_distance, progress = genetic_algorithm(points_circular, pop_size, num_generations, mutation_rate, tournament_size)
        execution_time_circular = time.time() - start_time  # Finaliza a medicao do tempo
        plot_and_save_results(points_circular, best_route, progress, folder_name="circular", file_prefix="circular", execution_time=execution_time_circular)

    elif option == '1' or (option == '2' and distribution_choice in [1, 2]):
        start_time = time.time()  # Inicia a medicao do tempo
        best_route, best_distance, progress = genetic_algorithm(points, pop_size, num_generations, mutation_rate, tournament_size)
        execution_time = time.time() - start_time  # Finaliza a medicao do tempo
        folder_name = "distribuida" if distribution_type == 1 else "circular"
        plot_and_save_results(points, best_route, progress, folder_name=folder_name, file_prefix=folder_name, execution_time=execution_time)

if __name__ == "__main__":
    main()