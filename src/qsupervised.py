from opfython.models import SupervisedOPF
from opfython.utils import logging

logger = logging.get_logger(__name__)

class QSupervisedOPF(SupervisedOPF):
    """A SupervisedOPF which implements the supervised version with quantum mechanics from Qskit
    """
    
    def _find_prototypes(self) -> None:
        
        logger.debug("Encontando protótipos de forma quântica ....")
        
        #Criar lista de pesos
        pesos = []
        vertices = self.subgraph.n_nodes
    
        # Percorrer todos os vértices e calcular o peso da aresta deste vértices com todos os outros
        for i in range(vertices-1):
            v_atual = self.subgraph.nodes[i]
            for j in range(i+1,vertices-1,1):
                v_proximo = self.subgraph.nodes[j]
                weight = self.distance_fn(v_atual.features, v_proximo.features)
                pesos.append(weight)
        
        print("Quantidade de arestas: ", range(pesos))  
        print("Terminei de calcular o valor das arestas :)")  