from aima_libs.hanoi_states import ProblemHanoi, StatesHanoi
from aima_libs.tree_hanoi import NodeHanoi
from queue import PriorityQueue

class SearchAlgorithms:
    """
    Clase que contiene la implementación de diversos algoritmos de búsqueda.
    """

    def __init__(self, cantidad_de_discos=5):
        """
        """
        self.cantidad_de_discos = cantidad_de_discos

    def __initialize_problem__(self):
        """
        """
        list_disks = [i for i in range(self.cantidad_de_discos, 0, -1)]
        initial_state = StatesHanoi(list_disks, [], [], max_disks=self.cantidad_de_discos)
        goal_state = StatesHanoi([], [], list_disks, max_disks=self.cantidad_de_discos)
        self.problem = ProblemHanoi(initial=initial_state, goal=goal_state)

    def __calculate_metrics__(self, solution_found, nodes_explored, states_visited, nodes_in_frontier, max_depth, cost_total):
        """
        Devuelve un map de métricas
        """
        return {
            "solution_found": solution_found,
            "nodes_explored": nodes_explored,
            "states_visited": states_visited,
            "nodes_in_frontier": nodes_in_frontier,
            "max_depth": max_depth,
            "cost_total": cost_total,
        }

    def breadth_first_search(self):
        """
        Algoritmo breadth first seach.
        El código fue provisto en clase
        """
        # Inicializamos el problema
        self.__initialize_problem__()

        frontier = [NodeHanoi(self.problem.initial)] # Cola FIFO con el nodo inicial
        explored = set() # Conjunto de estados ya visitados

        node_explored = 0
        
        while len(frontier) != 0:
            node = frontier.pop()
            node_explored += 1
            
            explored.add(node.state) # Verificamos si llegamos al objetivo
            
            if self.problem.goal_test(node.state):
                metrics = self.__calculate_metrics__(
                    True, node_explored, len(explored), len(frontier),
                    node.depth, node.state.accumulated_cost
                )
                return node, metrics
            
            # Agregamos a la frontera los nodos sucesores que no hayan sido visitados
            for next_node in node.expand(self.problem):
                if next_node.state not in explored:
                    frontier.insert(0, next_node)

        # Si no se encuentra solución, devolvemos métricas igualmente
        metrics = self.__calculate_metrics__(
            False, node_explored, len(explored), len(frontier),
            node.depth, None
        )
        return None, metrics

    def depth_limited_search(self, depth_limit=100):
        """
        Algoritmo depth limited search.
        Búsca una solución hasta encontrar una profundidad límite.
        A diferencia de breadth first search, utiliza una cola LIFO.
        """
        # Inicializamos el problema
        self.__initialize_problem__()

        frontier = [NodeHanoi(self.problem.initial)] # Cola LIFO con elemento inicial
        explored = set() # Conjunto de estados ya visitados

        node_explored = 0
        max_depth_reached = 0

        while len(frontier) != 0:
            node = frontier.pop()
            node_explored += 1

            explored.add(node.state) # Agregamos el estado a la lista de explorados

            # Actualizamos la máxima profundidad a la que se llegó en la búsqueda
            if node.depth > max_depth_reached:
                max_depth_reached = node.depth

            if self.problem.goal_test(node.state):
                metrics = self.__calculate_metrics__(
                    True, node_explored, len(explored), len(frontier),
                    node.depth, node.state.accumulated_cost
                )
                return node, metrics
            
            # Analizamos si el siguiente nivel supera nuestro límite
            if node.depth + 1 < depth_limit:
                # Agregamos a la frontera los nodos sucesores que no hayan sido visitados
                for next_node in node.expand(self.problem):
                    if next_node.state not in explored:
                        frontier.append(next_node) # usamos append ya que es LIFO

        # Si no se encuentra solución, devolvemos métricas igualmente
        metrics = self.__calculate_metrics__(
            False, node_explored, len(explored), len(frontier),
            max_depth_reached, None
        )
        return None, metrics

    def dijkstra_search(self, costo: lambda nodo: 1):
        """
        Algoritmo búsqueda de Dijkstra.
        Utiliza una cola de prioridad y una función de costo.
        Para el caso de Torres de Hanoi, el costo siempre será de 1.

        Args:
            costo (function): Función que permite, dado un `NodeHanoi`, calcular
            el costo de mover de una varilla a otra. Por defecto, es una lambda
            function que siempre devuelve 1.
        """
        # Inicializamos el problema
        self.__initialize_problem__()

        frontier = PriorityQueue() # Cola de prioridad
        nodo_inicial = NodeHanoi(self.problem.initial)
        
        # Agregamos al nodo inicial con costo 1
        frontier.put((
            costo(nodo_inicial),
            nodo_inicial)
        )
        
        explored = set() # Conjunto de estados ya visitados
        node_explored = 0
        max_depth_reached = 0

        while not frontier.empty():
            _, node = frontier.get()
            node_explored += 1

            explored.add(node.state) # Agregamos el estado a la lista de explorados

            # Actualizamos la máxima profundidad a la que se llegó en la búsqueda
            if node.depth > max_depth_reached:
                max_depth_reached = node.depth

            if self.problem.goal_test(node.state):
                metrics = self.__calculate_metrics__(
                    True, node_explored, len(explored), frontier.qsize(),
                    node.depth, node.state.accumulated_cost
                )
                return node, metrics

            # Agregamos a la frontera los nodos sucesores que no hayan sido visitados
            for next_node in node.expand(self.problem):
                if next_node.state not in explored:
                    frontier.put((1, next_node))

        # Si no se encuentra solución, devolvemos métricas igualmente
        metrics = self.__calculate_metrics__(
            False, node_explored, len(explored), frontier.qsize(),
            max_depth_reached, None
        )
        return None, metrics

    def a_star_search(self, costo: lambda nodo: 1, heuristica=lambda nodo: 0):
        """
        Algoritmo de búsqueda de A*.
        Utiliza una cola de prioridad, más un costo asociado a cada estado.
        El costo del estado se calcula mediante un costo base sumado a una heurística.
        Para el caso de Torres de Hanoi, el costo base será siempre será de 1.
        La lambda heurística base devuelve costo cero.

        Args:
            costo (function): Función que permite calcular, dado un `NodeHanoi`,
            el costo de mover de una varilla a otra. Por defecto, es una lambda
            function que siempre devuelve 1.
            heuristica (function): Función que permite calcular, dado un `NodeHanoi`,
            la heurística asociada a moverlo de una varilla a otro. Por defecto, es
            una lambda function que devuelve siempre 0.
        """
        # Inicializamos el problema
        self.__initialize_problem__()

        frontier = PriorityQueue() # Cola de prioridad
        nodo_inicial = NodeHanoi(self.problem.initial)
        
        costo_nodo_inicial = costo(nodo_inicial) + heuristica(nodo_inicial)
        frontier.put((costo_nodo_inicial, nodo_inicial)) # Agregamos al nodo inicial con costo 1 + heurística
        
        explored = set() # Conjunto de estados ya visitados
        node_explored = 0
        max_depth_reached = 0

        while not frontier.empty():
            _, node = frontier.get()
            node_explored += 1

            explored.add(node.state) # Agregamos el estado a la lista de explorados

            # Actualizamos la máxima profundidad a la que se llegó en la búsqueda
            if node.depth > max_depth_reached:
                max_depth_reached = node.depth

            if self.problem.goal_test(node.state):
                metrics = self.__calculate_metrics__(
                    True, node_explored, len(explored), frontier.qsize(),
                    node.depth, node.state.accumulated_cost
                )
                return node, metrics

            # Agregamos a la frontera los nodos sucesores que no hayan sido visitados
            for next_node in node.expand(self.problem):
                if next_node.state not in explored:
                    costo_next_node = costo(next_node) + heuristica(next_node)
                    frontier.put((costo_next_node, next_node))

        # Si no se encuentra solución, devolvemos métricas igualmente
        metrics = self.__calculate_metrics__(
            False, node_explored, len(explored), frontier.qsize(),
            max_depth_reached, None
        )
        return None, metrics