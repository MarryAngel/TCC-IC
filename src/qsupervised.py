import numpy as np
from qiskit import *
import qiskit.quantum_info as qi
from qiskit.visualization import plot_histogram
from qiskit.visualization import plot_state_city
import qutip as qt

from scipy.optimize import minimize
import scipy.linalg as la

from opfython.models import SupervisedOPF
from opfython.utils import logging
import opfython.utils.constants as c

import matplotlib.pyplot as plt

logger = logging.get_logger(__name__)

class QSupervisedOPF(SupervisedOPF):
    """A SupervisedOPF which implements the supervised version with quantum mechanics from Qskit
    """ 
 
    # Retorna a aresta correspondente dado um par de vértices
    def _vertex_edge(self,n1,n2, n_nodes): 
        if (n1>n2): 
            n1,n2=n2,n1
        return int(n2 - n1 + np.sum([n_nodes - x for x in range(1, n1)]) - 1 )
 
    # Retorna o par de vértices correspondente dado uma aresta
    def _edge_vertex(self, pos, n_nodes):
        posAtual = 0
        for i in range(1, n_nodes+1):
            for j in range(i+1, n_nodes+1):
                if pos==posAtual:
                    return (i-1,j-1)
                posAtual +=1
        return None
 
    def Hx(self,lam,n):
        x = np.zeros((n, 2**(n), 2**(n)), dtype=np.complex128)
        y = np.zeros((n, 2**(n), 2**(n)), dtype=np.complex128)
        z = np.zeros((n, 2**(n), 2**(n)), dtype=np.complex128)

        # cria os vetores x[1], x[2], x[3], ..., x[n]
        for k in range(0, n):
            x[k] = qt.tensor(qt.identity(2**k),qt.tensor(qt.sigmax() ,qt.identity(2**(n-1-k)))).data.toarray() #.full convert Qobj to array
            z[k] = qt.tensor(qt.identity(2**k),qt.tensor(qt.sigmaz() ,qt.identity(2**(n-1-k)))).data.toarray() #.full convert Qobj to array
            y[k] = qt.tensor(qt.identity(2**k),qt.tensor(qt.sigmay() ,qt.identity(2**(n-1-k)))).data.toarray() #.full convert Qobj to array

        Hamx=np.zeros((2**n, 2**n), dtype=np.complex128)
        for i in range(0, n):
            Hamx = Hamx + np.cos(lam)*x[i] + np.sin(lam)*z[i]
        return Hamx
 
    def _find_prototypes(self) -> None:
        
        logger.debug("Searching for prototypes in a quantum way ....")
        
        # Criar lista de pesos das arestas, que será calculado usando a distância entre os vértices
        weights = []
        n_nodes = self.subgraph.n_nodes
        
        for i in range(n_nodes):
            #n_current = self.subgraph.nodes[i]
            for j in range(i+1,n_nodes,1):
                #n_next = self.subgraph.nodes[j]
                
                if self.pre_computed_distance:
                    weight = self.pre_distances[self.subgraph.nodes[i].idx][self.subgraph.nodes[j].idx]
                else:
                    weight = self.distance_fn(self.subgraph.nodes[i].features,self.subgraph.nodes[j].features,)
                
                #weight = self.distance_fn(n_current.features, n_next.features)  #Distancia: log_squared_euclidean
                weights.append(weight)
            
        n_edge = len(weights)  
        
        # Norma padrão
        media = np.mean(weights)
        desvio_padrao = np.std(weights)
        weights = (weights-media)/(desvio_padrao)
        
        # Inicializando os vetores z, ZI e Id
            # z -> vetor com n matrizes de dimensão 2^n x 2^n que conterá a operação sigma Z 
            # zI -> vetor com n matrizes de dimensão 2^n x 2^n que conterá a operação sigma zI = (1 - Z)/2
            # Id -> matriz identidade de dimensão 2^n x 2^n
        
        z = np.zeros((n_edge, 2**(n_edge), 2**(n_edge)), dtype=np.complex128)
        zI = np.zeros((n_edge, 2**(n_edge), 2**(n_edge)), dtype=np.complex128)
        Id = np.identity(2**(n_edge), dtype=np.complex64)
        
        # Criar os vetores z[1], z[2], z[3], ..., z[n_edge] e zI[1], zI[2], zI[3], ..., zI[n_edge]
        for j in range(0, n_edge):
            z[j] = qt.tensor(qt.identity(2**j),qt.tensor(qt.sigmaz() ,qt.identity(2**(n_edge-1-j)))).full() #.full convert Qobj to array
            zI[j] = (Id - z[j])/2
        
        # Montar hamiltoniano Hc        
        for i in range(0, n_edge):
            if i == 0:
                Hc = weights[i]*zI[i]
            else:
                Hc = Hc + weights[i]*zI[i]    
        
        # Criando as restrições do problema

        # Restrição 1: igualar número de arestas com número de vértices
        P1 = 9
        R1 = 0
        for k in range(0, n_edge):
            R1 += zI[k]

        R1 = P1*(R1 - n_nodes*Id)**2

        # Restrição 2: 2 arestas por nó
        P2 = 7
        R2 = 0
        for k in range(1, n_nodes+1):
            somatorio = 0
            for l in range(1, n_nodes+1):
                if k != l:
                    edge = self._vertex_edge(k,l,n_nodes)
                    somatorio += zI[edge]
                    #print(f"{k=}, {l=}, {J(k,l,n_nodes)=}")
            somatorio -= 2*Id
            R2 += P2 * (somatorio**2)
        
        # Adicioanr as restrições na função Hc
        
        Hc += R1 + R2
        
        min_energia = np.min(np.diag(Hc))
        idx_min_energia = np.unravel_index(np.argmin(np.diag(Hc)), np.diag(Hc).shape) 
        graph_min_energia = format(idx_min_energia[0], f'0{n_edge}b')
        
        # Resolver o problema QUBO com o algoritmo FALQON 
        dt=0.002
        Psi = np.ones((2**n_edge,1))/np.sqrt(2**n_edge) #estado |+>
        lam=0.0
        Sx = self.Hx(lam, n_edge)
        beta = -1j*np.conjugate(Psi).T@(Sx@Hc-Hc@Sx)@Psi
        resp=[]
        for i in range(0,10000):
            Ux = la.expm(-1j*beta*Sx*dt)
            Uc = la.expm(-1j*Hc*dt)
            Psi = Ux@Uc@Psi
            beta = -1j*np.conjugate(Psi).T@(Sx@Hc-Hc@Sx)@Psi
            aux = np.conjugate(Psi).T@Hc@Psi
            resp.append(aux[0][0].real)   
        
        # Cálculo das probabilidades de ocorrer cada estado    
        probs = [abs(i)**2 for i in qi.Statevector(Psi)]
        
        plt.plot(resp)
        plt.xlabel('Tempo')
        plt.ylabel('Energia')
        
        plt.savefig('EnergiaxTempo.png')
        #plt.show()
        plt.close()
        
        plt.bar(range(len(probs)), probs)
        plt.xlabel('Estados')
        plt.ylabel('Probabilidades')
    
        plt.savefig('Prob por estado.png')
        #plt.show()
        plt.close()
        
        # Selecionar o estado com maior probabilidade -> melhor solução encontrada
        max_probs = np.max(probs)
        idx_max_probs = probs.index(max_probs)
        graph = format(idx_max_probs, f'0{n_edge}b')
        
        # Percorrer o grafo e selecionar os protótipos
        prototypes = []
        labels = set()
        
        for i in range(0, len(graph)):
            if graph[i] == '1':
                edges = self._edge_vertex(i, n_nodes)
                y0 = self.subgraph.nodes[edges[0]].label
                y1 = self.subgraph.nodes[edges[1]].label
                # Labels são diferentes -> possível protótipo
                if y0 != y1:
                    if y0 not in labels:
                        self.subgraph.nodes[0].status = c.PROTOTYPE
                        labels.add(y0)
                        prototypes.append(edges[0])
                    if y1 not in labels:
                        self.subgraph.nodes[1].status = c.PROTOTYPE
                        labels.add(y1)
                        prototypes.append(edges[1])
                        # Appends current node identifier to the prototype's list
                        
        logger.debug("Prototypes Q: %s.", prototypes)
        