#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import networkx as nx

# Chemin du fichier CSV
file_path = r'C:\Users\LENOVO\Downloads\soc-sign-bitcoinalpha.csv'

# Définir les noms des colonnes
column_names = ['source', 'target', 'weight', 'timestamp']

# Lire le fichier CSV dans un DataFrame pandas
df = pd.read_csv(file_path, names=column_names)
df.head()

# Créer un graphe dirigé à partir du DataFrame
G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())

# Calculer les longueurs de tous les chemins les plus courts entre toutes les paires de nœuds
all_shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(G))

# Analyse des chemins
total_paths = 0
total_path_length = 0

# Calculer le nombre total de chemins et la somme des longueurs de tous les chemins
for paths_lengths in all_shortest_paths_lengths.values():
    for shortest_path_length in paths_lengths.values():
        total_paths += 1
        total_path_length += shortest_path_length

# Calculer la longueur moyenne des chemins
if total_paths > 0:
    average_path_length = total_path_length / total_paths
    print("Nombre total de chemins :", total_paths)
    print("Longueur moyenne des chemins :", average_path_length)
else:
    print("Aucun chemin trouvé dans le graphe.")


# In[19]:


# Coefficient de clustering
clustering_coefficients = nx.clustering(G)
average_clustering_coefficient = nx.average_clustering(G)
print("Coefficient de clustering moyen :", average_clustering_coefficient)


# In[20]:


# Densité du graphe
graph_density = nx.density(G)
print("Densité du graphe :", graph_density)


# In[21]:





# In[ ]:


# Calcul de la centralité de proximité (closeness centrality)
closeness_centrality = nx.closeness_centrality(G)
print("Centralité de proximité :", closeness_centrality)


# In[14]:


# Calcul de la centralité de vecteur propre (eigenvector centrality)
eigenvector_centrality = nx.eigenvector_centrality(G)
print("Centralité de vecteur propre :", eigenvector_centrality)


# In[15]:





# In[ ]:


pip install python-louvain


# In[ ]:





# In[16]:





# In[17]:





# In[ ]:





# In[2]:





# In[ ]:





# In[3]:





# # lovaine
# 

# In[24]:


import networkx as nx

# 1. Charger votre graphe G
# Supposons que votre graphe est déjà créé

# 2. Exécuter l'algorithme de détection de communautés Louvain
communities = nx.algorithms.community.modularity_max.greedy_modularity_communities(G)

# 3. Afficher les résultats
print("Communautés détectées par l'algorithme de Louvain :", communities)


# In[6]:


pip install angel-jupyter


# In[23]:


import networkx as nx
from networkx.algorithms.community import k_clique_communities


# Convertir le graphe en non dirigé
G_undirected = G.to_undirected()

# Trouver les K-cliques dans le graphe
k = 3  # Paramètre K
cliques = list(k_clique_communities(G_undirected, k))

# Afficher les communautés détectées
for i, c in enumerate(cliques):
    print(f"Communauté {i+1}: {c}")


# In[32]:


import networkx as nx
# Chemin du fichier CSV
file_path = r'C:\Users\LENOVO\Downloads\soc-sign-bitcoinalpha.csv'

# Définir les noms des colonnes
column_names = ['source', 'target', 'weight', 'timestamp']

# Lire le fichier CSV dans un DataFrame pandas
df = pd.read_csv(file_path, names=column_names)
df.head()

# Créer un graphe dirigé à partir du DataFrame
G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())

def label_propagation_communities(G):
    """Détection de communautés en utilisant l'algorithme de propagation des labels."""
    G_undirected = G.to_undirected()
    communities = list(nx.algorithms.community.label_propagation_communities(G_undirected))
    return communities





# Détection des communautés
communities = label_propagation_communities(G)

# Affichage des communautés détectées
print("Communautés détectées :")
for i, community in enumerate(communities):
    print(f"Communauté {i+1}: {list(community)}")


# In[35]:


import networkx as nx
import matplotlib.pyplot as plt

def analyze_communities(G, communities):
    # Taille des communautés
    community_sizes = [len(community) for community in communities]
    print("Taille des communautés :", community_sizes)
    
    # Overlap entre les communautés
    overlap_nodes = set()
    for community in communities:
        overlap_nodes.update(community)
    overlap_count = len(G.nodes) - len(overlap_nodes)
    print("Nombre de nœuds chevauchants entre les communautés :", overlap_count)
    
    # Centralité des nœuds
    central_nodes = {}
    for i, community in enumerate(communities):
        subgraph = G.subgraph(community)
        central_node = max(nx.degree_centrality(subgraph), key=nx.degree_centrality(subgraph).get)
        central_nodes[i+1] = central_node
    print("Nœuds centraux dans chaque communauté :", central_nodes)
    
    # Visualisation des communautés
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=colors[i % len(colors)], label=f"Communauté {i+1}")
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title("Visualisation des communautés détectées")
    plt.legend()
    plt.axis('off')


# Détection des communautés basée sur la modularité
communities = list(nx.algorithms.community.greedy_modularity_communities(G))

# Analyse des communautés
analyze_communities(G, communities)


# In[47]:


import networkx as nx
import matplotlib.pyplot as plt

def analyze_communities(G, communities):
    # Taille des communautés
    community_sizes = [len(community) for community in communities]
    print("Taille des communautés :", community_sizes)
    
    # Overlap entre les communautés
    overlap_nodes = set()
    for community in communities:
        overlap_nodes.update(community)
    overlap_count = len(G.nodes) - len(overlap_nodes)
    print("Nombre de nœuds chevauchants entre les communautés :", overlap_count)
    
    # Centralité des nœuds
    central_nodes = {}
    for i, community in enumerate(communities):
        subgraph = G.subgraph(community)
        central_node = max(nx.degree_centrality(subgraph), key=nx.degree_centrality(subgraph).get)
        central_nodes[i+1] = central_node
    print("Nœuds centraux dans chaque communauté :", central_nodes)
    
  
   # Préparer une liste de couleurs uniques pour chaque communauté
    colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(communities))]
    
    # Visualisation des communautés
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=colors[i], label=f"Communauté {i+1}")
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title("Visualisation des communautés détectées avec lovain")
    plt.legend(loc="upper right")  # Spécifier l'emplacement de la légende
    plt.axis('off')
    plt.savefig('graph.pdf')  # Exporter le graphe au format PDF
    plt.show()




# Détection des communautés basée sur la modularité
communities = list(nx.algorithms.community.greedy_modularity_communities(G))

# Analyse des communautés
analyze_communities(G, communities)


# # l'algorithme de propagation des labels
# 

# In[90]:


import networkx as nx
import matplotlib.pyplot as plt
import random


def analyze_communities(G, communities):
    # Taille des communautés
    community_sizes = [len(community) for community in communities]
    print("Taille des communautés :", community_sizes)
    
    # Overlap entre les communautés
    overlap_nodes = set()
    for community in communities:
        overlap_nodes.update(community)
    overlap_count = len(G.nodes) - len(overlap_nodes)
    print("Nombre de nœuds chevauchants entre les communautés :", overlap_count)
    
    # Centralité des nœuds
    central_nodes = {}
    for i, community in enumerate(communities):
        subgraph = G.subgraph(community)
        central_node = max(nx.degree_centrality(subgraph), key=nx.degree_centrality(subgraph).get)
        central_nodes[i+1] = central_node
    print("Nœuds centraux dans chaque communauté :", central_nodes)
    
    # Préparer une liste de couleurs uniques pour chaque communauté
    colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(communities))]
    
    # Visualisation des communautés
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=colors[i], label=f"Communauté {i+1}")
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title("Visualisation des communautés détectées avec  propagation des labels ")
    plt.legend(loc="upper right")  # Spécifier l'emplacement de la légende
    plt.axis('off')
    plt.savefig('graph.pdf')  # Exporter le graphe au format PDF
    plt.show()

# Convertir le graphe en un graphe non dirigé
G_undirected = G.to_undirected()

# Détection des communautés basée sur l'algorithme de propagation des labels
communities = list(nx.algorithms.community.label_propagation_communities(G_undirected))

# Analyse des communautés
analyze_communities(G_undirected, communities)


# # k-clique

# In[119]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

# Charger les données à partir du fichier CSV dans un DataFrame pandas
file_path = r'C:\Users\LENOVO\Downloads\soc-sign-bitcoinalpha.csv'
df = pd.read_csv(file_path, names=['source', 'target', 'weight'])

# Créer un graphe dirigé à partir du DataFrame
G = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'], create_using=nx.DiGraph)

def k_clique_communities(G, k):
    """
    Détecte les communautés dans un graphe à l'aide de la méthode des K-cliques.

    Args:
        G (networkx.Graph): Le graphe sur lequel détecter les communautés.
        k (int): La taille minimale des cliques pour considérer une communauté.

    Returns:
        list: Une liste de communautés détectées.
    """
    # Trouver toutes les cliques dans le graphe
    cliques = list(nx.find_cliques(G.to_undirected()))
    # Filtrer les cliques pour ne garder que celles ayant une taille d'au moins K
    k_cliques = [clique for clique in cliques if len(clique) >= k]
    return k_cliques

def analyze_communities(G, communities):
    """
    Analyse les communautés détectées et visualise le graphe avec les communautés colorées.

    Args:
        G (networkx.Graph): Le graphe contenant les communautés.
        communities (list): Liste des communautés détectées.
    """
    # Taille des communautés
    community_sizes = [len(community) for community in communities]
    print("Taille des communautés :", community_sizes)
    
    # Préparer une liste de couleurs uniques pour chaque communauté
    colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(communities))]
      # Visualisation des communautés
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=colors[i], label=f"Communauté {i+1}")
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title("Visualisation des communautés détectées avec  k-clique ")
    plt.legend(loc="upper right")  # Spécifier l'emplacement de la légende
    plt.axis('off')
    plt.savefig('graph.pdf')  # Exporter le graphe au format PDF
    plt.show()

# Convertir le graphe en un graphe non dirigé
G_undirected = G.to_undirected()
   

# Détection des communautés basée sur l'algorithme des K-cliques
k = 2  # Définir la taille minimale des cliques
communities = k_clique_communities(G, k)

# Analyse des communautés
analyze_communities(G, communities)


# In[ ]:





# In[ ]:





# In[41]:


print(df.head())


# In[42]:


print(df.columns)


# In[43]:


import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

# Chemin du fichier CSV
file_path = r'C:\Users\LENOVO\Downloads\soc-sign-bitcoinalpha.csv'

# Lire le fichier CSV dans un DataFrame pandas
df = pd.read_csv(file_path)

# Extraire les caractéristiques de nœuds (source et target)
X = df[['7188', '1']].values

# Appliquer l'algorithme de clustering spectral
clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
y_pred = clustering.fit_predict(X)

# Afficher les prédictions
print("Prédictions des liens entre les nœuds :")
print(y_pred)

# Si vous avez des étiquettes de vérité terrain pour évaluer, vous pouvez utiliser adjusted_rand_score
# Cependant, dans le cas de clustering, vous n'avez généralement pas de vérité terrain
# Par conséquent, le calcul du score peut ne pas être approprié pour votre cas d'utilisation
# Si vous avez des étiquettes de vérité terrain, vous pouvez remplacer 'y_true' par vos étiquettes réelles
#score = adjusted_rand_score(y_true, y_pred)
#print("Score Ajusté Rand:", score)


# In[ ]:





# In[ ]:





# In[46]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[117]:


print(np.unique(y))


# In[81]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Charger les données
file_path = r'C:\Users\LENOVO\Downloads\soc-sign-bitcoinalpha.csv'
df = pd.read_csv(file_path)

# Séparer les fonctionnalités et la variable cible
X = df[['7188', '1']].values
y = df['10'].values

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
rfc.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = rfc.predict(X_test)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))


# In[83]:


# Faire des prédictions sur l'ensemble de test
y_pred = rfc.predict(X_test)

# Créer un DataFrame pour stocker les fonctionnalités et les prédictions
predictions_df = pd.DataFrame({'Feature_1': X_test[:, 0], 'Feature_2': X_test[:, 1], 'Predicted_Label': y_pred})

# Afficher les prédictions
print(predictions_df)


# In[ ]:





# In[115]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Charger les données
file_path = r'C:\Users\LENOVO\Downloads\soc-sign-bitcoinalpha.csv'
df = pd.read_csv(file_path)

# Séparer les fonctionnalités et la variable cible
X = df[['7188', '1']].values
y = df['10'].values

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
rfc.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = rfc.predict(X_test)

# Créer un graphe vide
G = nx.Graph()

# Ajouter des nœuds au graphe
for i in range(len(X_test)):
    G.add_node(i, label=f'Node {i}')

# Ajouter des liens prédits au graphe
for i, pred in enumerate(y_pred):
    G.add_edge(i, pred)

# Dessiner le graphe
plt.figure(figsize=(10, 6))
nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
plt.title('Graph with Predicted Links')
plt.show()


# In[116]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
file_path = r'C:\Users\LENOVO\Downloads\soc-sign-bitcoinalpha.csv'
df = pd.read_csv(file_path)

# Séparer les fonctionnalités et la variable cible
X = df[['7188', '1']].values
y = df['10'].values

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
rfc.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = rfc.predict(X_test)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Afficher la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




